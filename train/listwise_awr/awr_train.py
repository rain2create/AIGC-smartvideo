"""
awr_train.py  —  Listwise AWR 训练主循环（DDP）

用法：
  torchrun --nproc_per_node=2 train/listwise_awr/awr_train.py \
      --config train/listwise_awr/awr_v2_config.yaml

支持两种 loss（通过 config 中 loss_type 切换）：
  linear:  loss = -awr_coef · mean(A · δ) + β_anchor · mean(δ²)     ← 推荐
  clip:    loss = -min(surr1, surr2).mean() + β_anchor · mean(δ²)     ← PPO-style

支持从已有 LoRA 继续训练（config 中 init_lora_path 字段）：
  base → merge init_lora → 挂全新 LoRA → 训练
  reference = merged base（而非裸 base），delta 从 0 开始
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# 把项目根加入 path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.dpo.dpo_model import (
    compute_score,
    load_model_and_processor,
    load_model_with_merged_lora,
    save_lora,
)
from train.listwise_awr.awr_dataset import AWRDataCollator, AWRDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cfg = SimpleNamespace(**raw)

    # 路径解析：相对路径转为绝对路径
    for attr in ("output_dir", "log_dir", "splits_dir", "frame_cache_dir",
                 "groups_jsonl", "init_lora_path"):
        val = getattr(cfg, attr, None)
        if val and not Path(val).is_absolute():
            setattr(cfg, attr, str(REPO_ROOT / val))

    cfg.model_path = str(cfg.model_path)

    # 默认值
    if not hasattr(cfg, "loss_type"):
        cfg.loss_type = "clip"
    if not hasattr(cfg, "awr_coef"):
        cfg.awr_coef = 0.02

    return cfg


# ── AWR Loss ─────────────────────────────────────────────────────────────────

def awr_loss_linear(
    scores: torch.Tensor,        # [G] policy scores
    scores_ref: torch.Tensor,    # [G] reference scores
    rewards: torch.Tensor,       # [G] pre-computed rewards
    awr_coef: float = 0.02,
    beta_anchor: float = 0.01,
    norm_weight: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    Linear AWR loss: advantage-weighted regression + anchor（KL）。

    loss = norm_weight · (-awr_coef · mean(A · δ)) + β_anchor · mean(δ²)

    anchor 不受 weight 影响（trust region 不应因 hard group 放大）。
    """
    delta = scores - scores_ref                                    # [G]

    # z-score advantage
    advantage = (rewards - rewards.mean()) / (rewards.std() + eps)  # [G]

    # linear policy gradient
    pg_loss = -awr_coef * (advantage * delta).mean()

    # anchor/KL regularization（不加权）
    anchor_loss = beta_anchor * (delta.pow(2)).mean()

    total_loss = norm_weight * pg_loss + anchor_loss

    with torch.no_grad():
        metrics = {
            "pg_loss": pg_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "delta_mean": delta.mean().item(),
            "delta_std": delta.std().item(),
            "advantage_mean": advantage.mean().item(),
            "score_policy_mean": scores.mean().item(),
            "score_ref_mean": scores_ref.mean().item(),
            "score_raw_max": scores.abs().max().item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "group_size": len(scores),
            "norm_weight": norm_weight,
        }

    return total_loss, metrics


def awr_loss_clip(
    scores: torch.Tensor,        # [G] policy scores
    scores_ref: torch.Tensor,    # [G] reference scores
    rewards: torch.Tensor,       # [G] pre-computed rewards
    beta_anchor: float = 0.01,
    clip_eps: float = 0.2,
    norm_weight: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    PPO-style clipped AWR loss + anchor（KL）。

    loss = norm_weight · (-min(surr1, surr2).mean()) + β_anchor · mean(δ²)
    """
    delta = scores - scores_ref                                    # [G]

    # z-score advantage
    advantage = (rewards - rewards.mean()) / (rewards.std() + eps)  # [G]

    # PPO-style clipped surrogate
    ratio = torch.exp(delta)                                       # [G]
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    pg_loss = -torch.min(surr1, surr2).mean()

    # anchor/KL regularization（不加权）
    anchor_loss = beta_anchor * (delta.pow(2)).mean()

    total_loss = norm_weight * pg_loss + anchor_loss

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
        metrics = {
            "pg_loss": pg_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "delta_mean": delta.mean().item(),
            "delta_std": delta.std().item(),
            "ratio_mean": ratio.mean().item(),
            "clip_frac": clip_frac,
            "advantage_mean": advantage.mean().item(),
            "score_policy_mean": scores.mean().item(),
            "score_ref_mean": scores_ref.mean().item(),
            "score_raw_max": scores.abs().max().item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "group_size": len(scores),
            "norm_weight": norm_weight,
        }

    return total_loss, metrics


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def to_device(inputs_dict: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs_dict.items()
    }


def sync_trainable_gradients(model: torch.nn.Module, world_size: int) -> None:
    """Average trainable gradients across ranks without DDP autograd hooks."""
    if world_size <= 1:
        return
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


class TrainLogger:
    def __init__(self, log_dir: str, rank: int):
        self.rank = rank
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open(Path(log_dir) / f"train_log_rank{rank}.jsonl", "a")

    def log(self, step: int, **kwargs):
        entry = {"step": step, **kwargs}
        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train(cfg: SimpleNamespace):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0

    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(Path(cfg.log_dir) / f"train_rank{rank}.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)

    if is_main:
        logger.info("Config: %s", vars(cfg))
        logger.info("Loss type: %s", cfg.loss_type)

    # ── 模型 ──────────────────────────────────────────────────────────────
    init_lora = getattr(cfg, "init_lora_path", None)
    if init_lora:
        logger.info("Using merged LoRA init: %s", init_lora)
        model, processor, yes_id, no_id = load_model_with_merged_lora(cfg)
    else:
        model, processor, yes_id, no_id = load_model_and_processor(cfg)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()

    # ── 数据 ──────────────────────────────────────────────────────────────
    groups_path = getattr(cfg, "groups_jsonl", None)
    if not groups_path:
        groups_path = str(Path(cfg.splits_dir) / "train_groups.jsonl")
    logger.info("Groups file: %s", groups_path)

    train_dataset = AWRDataset(
        groups_jsonl_path=groups_path,
        frame_cache_dir=cfg.frame_cache_dir,
        max_frames=8,
    )
    collator = AWRDataCollator(
        processor=processor,
        instruction=cfg.instruction,
        total_pixels=cfg.total_pixels,
        max_seq_len=cfg.max_seq_len,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        train_dataset,
        batch_size=1,         # 一个 group = 一个 training unit
        sampler=sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # 计算 mean_weight 用于归一化（使加权后整体 loss scale 不变）
    mean_weight = sum(g.get("weight", 1.0) for g in train_dataset.groups) / len(train_dataset.groups)
    if is_main:
        logger.info("Mean group weight: %.3f (will normalize to 1.0)", mean_weight)

    # ── Optimizer + LR Schedule ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=0.01,
    )
    total_steps = cfg.max_steps
    warmup_steps = cfg.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 输出目录 ──────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_logger = TrainLogger(cfg.log_dir, rank)
    tb_writer = SummaryWriter(log_dir=cfg.log_dir) if is_main else None

    # ── loss 选择 ─────────────────────────────────────────────────────────
    loss_type = cfg.loss_type  # "linear" or "clip"

    # ── 训练主循环 ────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    accum_step = 0
    accum_loss = 0.0
    accum_metrics = {}
    optimizer.zero_grad()

    grad_accum = getattr(cfg, "grad_accum", 1)

    epoch = 0
    while global_step < total_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in loader:
            if global_step >= total_steps:
                break

            candidate_inputs = batch["candidate_inputs"]   # list of G dicts
            rewards = batch["rewards"].to(device)           # [G]
            group_weight = batch["weight"]                  # float
            if isinstance(group_weight, torch.Tensor):
                group_weight = group_weight.item()

            # 跳过 reward 方差为 0 的 group（z-score 无意义）
            if rewards.std() < 1e-6:
                continue

            # ── 逐候选前向 + backward ──
            # 不把整个 group 的 G 个候选计算图同时留在显存里。Qwen3-VL 8B + 视频
            # 输入下，G≈11 时一次性 stack 再 backward 很容易在第一个 group OOM。
            group_size = len(candidate_inputs)
            norm_weight = group_weight / mean_weight
            beta_anchor = getattr(cfg, "beta_anchor", 0.01)
            clip_eps = getattr(cfg, "clip_eps", 0.2)
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            pg_vals = []
            anchor_vals = []
            delta_vals = []
            ratio_vals = []
            clip_vals = []
            policy_vals = []
            ref_vals = []
            raw_abs_vals = []

            for cand_idx, cand_inputs in enumerate(candidate_inputs):
                cand_inputs = to_device(cand_inputs, device)

                # policy forward: 直接用 module，避免 DDP autograd hook 在 group 内
                # 多次 backward + 不同 group_size 时触发 reducer 断言。梯度同步在
                # optimizer step 前手动 all-reduce。
                s_policy = compute_score(model.module, cand_inputs, yes_id, no_id).squeeze(0)

                # reference forward
                model.module.disable_adapter_layers()
                with torch.no_grad():
                    s_ref = compute_score(model.module, cand_inputs, yes_id, no_id).squeeze(0)
                model.module.enable_adapter_layers()

                delta = s_policy - s_ref.detach()
                if loss_type == "linear":
                    pg_loss_i = -cfg.awr_coef * advantage[cand_idx] * delta
                    candidate_loss = norm_weight * pg_loss_i + beta_anchor * delta.pow(2)
                else:  # "clip"
                    ratio = torch.exp(delta)
                    surr1 = ratio * advantage[cand_idx]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage[cand_idx]
                    pg_loss_i = -torch.min(surr1, surr2)
                    candidate_loss = norm_weight * pg_loss_i + beta_anchor * delta.pow(2)

                (candidate_loss / group_size / grad_accum).backward()

                with torch.no_grad():
                    pg_vals.append(pg_loss_i.item())
                    anchor_vals.append((beta_anchor * delta.pow(2)).item())
                    delta_vals.append(delta.item())
                    policy_vals.append(s_policy.item())
                    ref_vals.append(s_ref.item())
                    raw_abs_vals.append(abs(s_policy.item()))
                    if loss_type == "clip":
                        ratio_val = ratio.item()
                        ratio_vals.append(ratio_val)
                        clip_vals.append(float(abs(ratio_val - 1.0) > clip_eps))

            pg_loss_mean = sum(pg_vals) / group_size
            anchor_loss_mean = sum(anchor_vals) / group_size
            loss_value = norm_weight * pg_loss_mean + anchor_loss_mean
            delta_mean = sum(delta_vals) / group_size
            delta_var = sum((x - delta_mean) ** 2 for x in delta_vals) / max(1, group_size - 1)
            loss_metrics = {
                "pg_loss": pg_loss_mean,
                "anchor_loss": anchor_loss_mean,
                "delta_mean": delta_mean,
                "delta_std": math.sqrt(delta_var),
                "advantage_mean": advantage.mean().item(),
                "score_policy_mean": sum(policy_vals) / group_size,
                "score_ref_mean": sum(ref_vals) / group_size,
                "score_raw_max": max(raw_abs_vals),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "group_size": group_size,
                "norm_weight": norm_weight,
            }
            if loss_type == "clip":
                loss_metrics["ratio_mean"] = sum(ratio_vals) / group_size
                loss_metrics["clip_frac"] = sum(clip_vals) / group_size

            accum_loss += loss_value
            accum_step += 1

            if global_step == 0 and accum_step <= grad_accum:
                logger.info(
                    "rank=%d  warmup group %d/%d done  G=%d  loss=%.4f  raw_max=%.2f",
                    rank, accum_step, grad_accum, group_size, loss_value, max(raw_abs_vals),
                )

            # 累加 micro-batch 指标
            if not accum_metrics:
                accum_metrics = {k: v for k, v in loss_metrics.items()}
                accum_metrics["_count"] = 1
            else:
                for k, v in loss_metrics.items():
                    if k == "score_raw_max":
                        accum_metrics[k] = max(accum_metrics.get(k, 0), v)
                    elif k == "group_size":
                        accum_metrics[k] = accum_metrics.get(k, 0) + v
                    else:
                        accum_metrics[k] = accum_metrics.get(k, 0) + v
                accum_metrics["_count"] += 1

            # ── optimizer step ──
            if accum_step % grad_accum == 0:
                sync_trainable_gradients(model.module, world_size)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                n = accum_metrics.pop("_count")
                avg = {}
                for k, v in accum_metrics.items():
                    if k == "score_raw_max":
                        avg[k] = v
                    elif k == "group_size":
                        avg[k] = v / n
                    else:
                        avg[k] = v / n
                avg_loss = accum_loss / grad_accum
                lr_now = scheduler.get_last_lr()[0]

                log_kwargs = dict(
                    step=global_step,
                    loss=round(avg_loss, 6),
                    lr=round(lr_now, 8),
                    pg_loss=round(avg["pg_loss"], 6),
                    anchor_loss=round(avg["anchor_loss"], 6),
                    delta_mean=round(avg["delta_mean"], 4),
                    delta_std=round(avg["delta_std"], 4),
                    advantage_mean=round(avg["advantage_mean"], 4),
                    score_policy_mean=round(avg["score_policy_mean"], 4),
                    score_ref_mean=round(avg["score_ref_mean"], 4),
                    score_raw_max=round(avg["score_raw_max"], 4),
                    reward_mean=round(avg["reward_mean"], 4),
                    reward_std=round(avg["reward_std"], 4),
                    avg_group_size=round(avg["group_size"], 1),
                    norm_weight=round(avg.get("norm_weight", 1.0), 3),
                    grad_norm=round(float(grad_norm), 6),
                )
                if "ratio_mean" in avg:
                    log_kwargs["ratio_mean"] = round(avg["ratio_mean"], 4)
                    log_kwargs["clip_frac"] = round(avg["clip_frac"], 4)
                train_logger.log(**log_kwargs)

                accum_loss = 0.0
                accum_metrics = {}

                # 日志格式
                extra = ""
                if "ratio_mean" in avg:
                    extra = f"  ratio={avg['ratio_mean']:.4f}  clip={avg['clip_frac']:.2f}"
                logger.info(
                    "rank=%d  step=%d  loss=%.4f  pg=%.4f  anch=%.4f  "
                    "δ_mean=%.4f  δ_std=%.4f%s  "
                    "adv=%.4f  s_π=%.4f  s_ref=%.4f  raw_max=%.2f  "
                    "r_mean=%.2f  r_std=%.2f  G=%.0f  w=%.2f  grad=%.4f  lr=%.2e",
                    rank, global_step, avg_loss,
                    avg["pg_loss"], avg["anchor_loss"],
                    avg["delta_mean"], avg["delta_std"], extra,
                    avg["advantage_mean"],
                    avg["score_policy_mean"], avg["score_ref_mean"],
                    avg["score_raw_max"],
                    avg["reward_mean"], avg["reward_std"],
                    avg["group_size"], avg.get("norm_weight", 1.0), float(grad_norm), lr_now,
                )

                if is_main and tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, global_step)
                    tb_writer.add_scalar("train/pg_loss", avg["pg_loss"], global_step)
                    tb_writer.add_scalar("train/anchor_loss", avg["anchor_loss"], global_step)
                    tb_writer.add_scalar("train/delta_mean", avg["delta_mean"], global_step)
                    tb_writer.add_scalar("train/delta_std", avg["delta_std"], global_step)
                    tb_writer.add_scalar("train/advantage_mean", avg["advantage_mean"], global_step)
                    tb_writer.add_scalar("train/score_policy_mean", avg["score_policy_mean"], global_step)
                    tb_writer.add_scalar("train/score_ref_mean", avg["score_ref_mean"], global_step)
                    tb_writer.add_scalar("train/score_raw_max", avg["score_raw_max"], global_step)
                    tb_writer.add_scalar("train/reward_mean", avg["reward_mean"], global_step)
                    tb_writer.add_scalar("train/lr", lr_now, global_step)
                    tb_writer.add_scalar("train/avg_group_size", avg["group_size"], global_step)
                    tb_writer.add_scalar("train/norm_weight", avg.get("norm_weight", 1.0), global_step)
                    tb_writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    if "ratio_mean" in avg:
                        tb_writer.add_scalar("train/ratio_mean", avg["ratio_mean"], global_step)
                        tb_writer.add_scalar("train/clip_frac", avg["clip_frac"], global_step)

                if is_main and global_step % cfg.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    save_lora(model.module, ckpt_dir)
                    logger.info("Checkpoint saved: %s", ckpt_dir)

    # ── 最终保存 ──────────────────────────────────────────────────────────
    if is_main:
        save_lora(model.module, output_dir / "final")
        logger.info("Training complete. Final adapter: %s/final", output_dir)
        train_logger.close()
        if tb_writer:
            tb_writer.close()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to awr config yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
