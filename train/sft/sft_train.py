"""
sft_train.py  —  SFT 训练主循环（DDP）

用法：
  torchrun --nproc_per_node=2 train/sft/sft_train.py \
      --config train/sft/sft_config.yaml

SFT Loss（pointwise BCE 纠错）：
    score = logit_yes - logit_no
    loss  = BCEWithLogitsLoss(score, target)
    正样本 target=1 → 推高 score
    负样本 target=0 → 压低 score
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

from train.dpo.dpo_model import compute_score, load_model_and_processor, save_lora
from train.sft.sft_dataset import SFTDataCollator, SFTDataset

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

    for attr in ("output_dir", "log_dir", "sft_data_path", "frame_cache_dir"):
        val = getattr(cfg, attr, None)
        if val and not Path(val).is_absolute():
            setattr(cfg, attr, str(REPO_ROOT / val))

    cfg.model_path = str(cfg.model_path)
    return cfg


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def to_device(inputs_dict: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs_dict.items()
    }


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

    # ── 模型 ──────────────────────────────────────────────────────────────
    model, processor, yes_id, no_id = load_model_and_processor(cfg)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── 数据 ──────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(
        sft_jsonl_path=cfg.sft_data_path,
        frame_cache_dir=cfg.frame_cache_dir,
        max_frames=8,
    )
    collator = SFTDataCollator(
        processor=processor,
        instruction=cfg.instruction,
        total_pixels=cfg.total_pixels,
        max_seq_len=cfg.max_seq_len,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        train_dataset,
        batch_size=getattr(cfg, "batch_size_per_gpu", 1),
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
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 输出目录 ──────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_logger = TrainLogger(cfg.log_dir, rank)
    tb_writer = SummaryWriter(log_dir=cfg.log_dir) if is_main else None

    # ── 训练主循环 ────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    accum_step = 0
    accum_loss = 0.0
    accum_correct = 0
    accum_total = 0
    accum_pos_score = 0.0
    accum_neg_score = 0.0
    accum_pos_count = 0
    accum_neg_count = 0
    optimizer.zero_grad()

    grad_accum = getattr(cfg, "grad_accum", 1)

    epoch = 0
    while global_step < total_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in loader:
            if global_step >= total_steps:
                break

            targets = batch.pop("targets").to(device)  # [B]
            inputs = to_device(batch, device)

            # ── Forward ──
            scores = compute_score(model, inputs, yes_id, no_id)  # [B]
            pos_weight = torch.tensor([getattr(cfg, "pos_weight", 1.0)], device=device)
            loss = F.binary_cross_entropy_with_logits(scores, targets, pos_weight=pos_weight)

            # ── 指标 ──
            with torch.no_grad():
                preds = (scores > 0).float()
                correct = (preds == targets).sum().item()
                accum_correct += correct
                accum_total += targets.numel()

                pos_mask = targets == 1
                neg_mask = targets == 0
                if pos_mask.any():
                    accum_pos_score += scores[pos_mask].sum().item()
                    accum_pos_count += pos_mask.sum().item()
                if neg_mask.any():
                    accum_neg_score += scores[neg_mask].sum().item()
                    accum_neg_count += neg_mask.sum().item()

            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            accum_loss += loss.item()
            accum_step += 1

            # ── Optimizer step ──
            if accum_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / grad_accum
                accuracy = accum_correct / max(accum_total, 1)
                pos_score_mean = accum_pos_score / max(accum_pos_count, 1)
                neg_score_mean = accum_neg_score / max(accum_neg_count, 1)
                lr_now = scheduler.get_last_lr()[0]

                train_logger.log(
                    step=global_step,
                    loss=round(avg_loss, 6),
                    lr=round(lr_now, 8),
                    accuracy=round(accuracy, 4),
                    pos_score_mean=round(pos_score_mean, 4),
                    neg_score_mean=round(neg_score_mean, 4),
                    pos_count=accum_pos_count,
                    neg_count=accum_neg_count,
                )

                logger.info(
                    "rank=%d  step=%d  loss=%.4f  acc=%.3f  "
                    "s_pos=%.3f  s_neg=%.3f  lr=%.2e",
                    rank, global_step, avg_loss, accuracy,
                    pos_score_mean, neg_score_mean, lr_now,
                )

                if is_main and tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, global_step)
                    tb_writer.add_scalar("train/accuracy", accuracy, global_step)
                    tb_writer.add_scalar("train/pos_score_mean", pos_score_mean, global_step)
                    tb_writer.add_scalar("train/neg_score_mean", neg_score_mean, global_step)
                    tb_writer.add_scalar("train/lr", lr_now, global_step)

                # reset accumulators
                accum_loss = 0.0
                accum_correct = 0
                accum_total = 0
                accum_pos_score = 0.0
                accum_neg_score = 0.0
                accum_pos_count = 0
                accum_neg_count = 0

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
    parser.add_argument("--config", required=True, help="Path to sft_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
