"""
dpo_train.py  —  DPO 训练主循环（DDP）

用法：
  torchrun --nproc_per_node=2 train/dpo/dpo_train.py \
      --config train/dpo/dpo_config.yaml

DPO Loss（score-based）：
    s_w, s_r   = policy  forward on chosen/rejected
    s_w_ref, s_r_ref = reference forward（disable_adapter_layers + no_grad）
    loss = -log σ( β · [(s_w - s_w_ref) - (s_r - s_r_ref)] )
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
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# 把项目根加入 path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.dpo.dpo_dataset import DPODataCollator, DPODataset
from train.dpo.dpo_model import compute_score, load_model_and_processor, save_lora

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

    # 相对路径解析（相对于项目根）
    for attr in ("output_dir", "log_dir", "splits_dir", "frame_cache_dir"):
        val = getattr(cfg, attr, None)
        if val and not Path(val).is_absolute():
            setattr(cfg, attr, str(REPO_ROOT / val))

    # model_path 保持原样（通常是绝对路径）
    cfg.model_path = str(cfg.model_path)

    return cfg


# ── DPO Loss ─────────────────────────────────────────────────────────────────

def dpo_loss(
    s_w: torch.Tensor,
    s_w_ref: torch.Tensor,
    s_r: torch.Tensor,
    s_r_ref: torch.Tensor,
    beta: float,
    beta_anchor: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    DPO loss（score-based）+ anchor 正则项。

    anchor 惩罚 policy 分数偏离 reference 太远，防止 LoRA 通过对所有输入
    加全局正偏置来规避 DPO 梯度（score explosion 问题，见 findings/score_explosion.md）。

    Returns:
        total_loss, metrics_dict（detached，供日志用）
    """
    delta_w = s_w - s_w_ref
    delta_r = s_r - s_r_ref
    adv = delta_w - delta_r

    logits = beta * adv
    preference_loss = -F.logsigmoid(logits).mean()
    anchor_loss = beta_anchor * (delta_w.pow(2) + delta_r.pow(2)).mean()
    total_loss = preference_loss + anchor_loss

    # detached 诊断指标
    with torch.no_grad():
        metrics = {
            "preference_loss": preference_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "delta_w": delta_w.mean().item(),
            "delta_r": delta_r.mean().item(),
            "adv_mean": adv.mean().item(),
            "adv_pos_rate": (adv > 0).float().mean().item(),
            "score_raw_max": max(s_w.abs().max().item(), s_r.abs().max().item()),
        }

    return total_loss, metrics


# ── 训练步骤 ─────────────────────────────────────────────────────────────────

def forward_one(model, inputs_dict: dict, yes_id: int, no_id: int) -> torch.Tensor:
    """把 DataCollator 输出的某一侧（chosen/rejected）送入 compute_score。"""
    return compute_score(model, inputs_dict, yes_id, no_id)


def to_device(inputs_dict: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs_dict.items()
    }


def split_chosen_rejected(batch: dict, device: torch.device):
    """从 DataCollator 的 batch 中拆出 chosen/rejected inputs 并移到 device。"""
    chosen, rejected = {}, {}
    for k, v in batch.items():
        if k.startswith("chosen_"):
            chosen[k[len("chosen_"):]] = v
        elif k.startswith("rejected_"):
            rejected[k[len("rejected_"):]] = v
    return to_device(chosen, device), to_device(rejected, device)


# ── 日志工具 ─────────────────────────────────────────────────────────────────

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
    # DDP 初始化
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0

    # 每个 rank 写自己的文本日志
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(Path(cfg.log_dir) / f"train_rank{rank}.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)

    if is_main:
        logger.info("Config: %s", vars(cfg))

    # ── 模型 ──────────────────────────────────────────────────────────────
    model, processor, yes_id, no_id = load_model_and_processor(cfg)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── 数据 ──────────────────────────────────────────────────────────────
    train_dataset = DPODataset(
        pairs_jsonl_path=Path(cfg.splits_dir) / "train_pairs.jsonl",
        frame_cache_dir=cfg.frame_cache_dir,
        max_frames=8,
    )
    collator = DPODataCollator(
        processor=processor,
        instruction=cfg.instruction,
        total_pixels=cfg.total_pixels,
        max_seq_len=cfg.max_seq_len,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_per_gpu,
        sampler=sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

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
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))  # cosine decay to 10%

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 输出目录 ──────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_logger = TrainLogger(cfg.log_dir, rank)
    tb_writer = SummaryWriter(log_dir=cfg.log_dir) if is_main else None

    # ── 训练主循环 ────────────────────────────────────────────────────────
    # global_step 计 optimizer steps（参数更新次数），与 max_steps / warmup_steps 单位一致
    model.train()
    global_step = 0   # optimizer step 计数
    accum_step  = 0   # batch 计数（用于 grad_accum 判断）
    accum_loss  = 0.0
    # 累加器：所有指标在 grad_accum 窗口内累加，记日志时取均值
    accum_metrics = {}
    optimizer.zero_grad()

    epoch = 0
    while global_step < total_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in loader:
            if global_step >= total_steps:
                break

            chosen_inputs, rejected_inputs = split_chosen_rejected(batch, device)

            # ── policy forward（raw score，不 clamp）──
            s_w = forward_one(model, chosen_inputs, yes_id, no_id)
            s_r = forward_one(model, rejected_inputs, yes_id, no_id)

            # ── reference forward（LoRA 失效，no_grad）──
            model.module.disable_adapter_layers()
            with torch.no_grad():
                s_w_ref = forward_one(model.module, chosen_inputs, yes_id, no_id)
                s_r_ref = forward_one(model.module, rejected_inputs, yes_id, no_id)
            model.module.enable_adapter_layers()

            loss, loss_metrics = dpo_loss(s_w, s_w_ref, s_r, s_r_ref, beta=cfg.beta, beta_anchor=getattr(cfg, "beta_anchor", 0.0))
            scaled_loss = loss / cfg.grad_accum
            scaled_loss.backward()
            accum_loss += loss.item()
            accum_step += 1

            # 累加 micro-batch 指标
            mb = {
                "s_w": s_w.mean().item(),
                "s_r": s_r.mean().item(),
                **loss_metrics,
            }
            if not accum_metrics:
                accum_metrics = {k: v for k, v in mb.items()}
                accum_metrics["score_raw_max"] = mb["score_raw_max"]
                accum_metrics["_count"] = 1
            else:
                for k, v in mb.items():
                    if k == "score_raw_max":
                        accum_metrics[k] = max(accum_metrics[k], v)
                    else:
                        accum_metrics[k] += v
                accum_metrics["_count"] += 1

            # ── optimizer step（每 grad_accum 个 batch）──
            if accum_step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1  # 只在真正更新参数后才 +1

                # ── 每个 rank 各自记录本卡指标（全部是 grad_accum 窗口均值）──
                n = accum_metrics.pop("_count")
                avg = {k: (v if k == "score_raw_max" else v / n) for k, v in accum_metrics.items()}
                avg_loss      = accum_loss / cfg.grad_accum
                s_w_mean_val  = avg["s_w"]
                s_r_mean_val  = avg["s_r"]
                reward_margin = s_w_mean_val - s_r_mean_val
                lr_now        = scheduler.get_last_lr()[0]
                train_logger.log(
                    step=global_step,
                    loss=round(avg_loss, 6),
                    lr=round(lr_now, 8),
                    s_w_mean=round(s_w_mean_val, 4),
                    s_r_mean=round(s_r_mean_val, 4),
                    reward_margin=round(reward_margin, 4),
                    preference_loss=round(avg["preference_loss"], 6),
                    anchor_loss=round(avg["anchor_loss"], 6),
                    delta_w=round(avg["delta_w"], 4),
                    delta_r=round(avg["delta_r"], 4),
                    adv_mean=round(avg["adv_mean"], 4),
                    adv_pos_rate=round(avg["adv_pos_rate"], 4),
                    score_raw_max=round(avg["score_raw_max"], 4),
                )
                accum_loss = 0.0
                accum_metrics = {}

                # ── 文本日志：两个 rank 各写各的 ──
                logger.info(
                    "rank=%d  step=%d  loss=%.4f  s_w=%.4f  s_r=%.4f  margin=%.4f  "
                    "adv=%.4f  adv_pos=%.2f  pref=%.4f  anch=%.4f  "
                    "dw=%.4f  dr=%.4f  raw_max=%.2f  lr=%.2e",
                    rank, global_step, avg_loss, s_w_mean_val, s_r_mean_val, reward_margin,
                    avg["adv_mean"], avg["adv_pos_rate"],
                    avg["preference_loss"], avg["anchor_loss"],
                    avg["delta_w"], avg["delta_r"],
                    avg["score_raw_max"], lr_now,
                )

                # ── TensorBoard：rank 0 only ──
                if is_main and tb_writer:
                    tb_writer.add_scalar("train/loss",          avg_loss,      global_step)
                    tb_writer.add_scalar("train/reward_margin", reward_margin, global_step)
                    tb_writer.add_scalar("train/s_w_mean",      s_w_mean_val,  global_step)
                    tb_writer.add_scalar("train/s_r_mean",      s_r_mean_val,  global_step)
                    tb_writer.add_scalar("train/lr",            lr_now,        global_step)
                    tb_writer.add_scalar("train/preference_loss", loss_metrics["preference_loss"], global_step)
                    tb_writer.add_scalar("train/anchor_loss",     loss_metrics["anchor_loss"],     global_step)
                    tb_writer.add_scalar("train/delta_w",         loss_metrics["delta_w"],         global_step)
                    tb_writer.add_scalar("train/delta_r",         loss_metrics["delta_r"],         global_step)
                    tb_writer.add_scalar("train/adv_mean",        loss_metrics["adv_mean"],        global_step)
                    tb_writer.add_scalar("train/adv_pos_rate",    loss_metrics["adv_pos_rate"],    global_step)
                    tb_writer.add_scalar("train/score_raw_max",   loss_metrics["score_raw_max"],   global_step)

                # ── checkpoint ──
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


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dpo_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
