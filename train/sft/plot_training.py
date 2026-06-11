"""
plot_training.py  —  从 train_log_rank{0,1}.jsonl 生成 SFT 训练曲线图

生成三张图：
  training_curves_rank0.png   — rank 0 本卡指标
  training_curves_rank1.png   — rank 1 本卡指标
  training_curves_mean.png    — 两卡按 step 对齐后的均值

用法：
  python train/sft/plot_training.py
  python train/sft/plot_training.py --log-dir train/logs/qwen_sft
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_log(log_file: Path):
    records = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_MEAN_KEYS = [
    "loss", "accuracy", "pos_score_mean", "neg_score_mean",
]


def compute_mean(r0, r1):
    steps0 = {r["step"]: r for r in r0}
    steps1 = {r["step"]: r for r in r1}
    common = sorted(set(steps0) & set(steps1))
    mean = []
    for step in common:
        a, b = steps0[step], steps1[step]
        rec = {"step": step, "lr": a.get("lr")}
        for k in _MEAN_KEYS:
            rec[k] = (a.get(k, 0) + b.get(k, 0)) / 2
        mean.append(rec)
    return mean


def _smooth(values, window=20):
    """简单移动平均平滑。"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return list(values[: window - 1]) + list(smoothed)


def plot(records, output_path: Path, title: str):
    steps = [r["step"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    # ── Panel 0: BCE Loss ──
    ax = axes_flat[0]
    losses = [r["loss"] for r in records]
    ax.plot(steps, losses, color="#e74c3c", alpha=0.3, linewidth=0.8)
    ax.plot(steps, _smooth(losses), color="#e74c3c", linewidth=1.5, label="smoothed")
    ax.axhline(y=0.6931, color="gray", linestyle="--", linewidth=0.8, label="log(2)")
    ax.set_title("BCE Loss")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 1: Accuracy ──
    ax = axes_flat[1]
    accs = [r.get("accuracy", 0) for r in records]
    ax.plot(steps, accs, color="#2ecc71", alpha=0.3, linewidth=0.8)
    ax.plot(steps, _smooth(accs), color="#2ecc71", linewidth=1.5, label="smoothed")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
    ax.set_title("Accuracy")
    ax.set_xlabel("step")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Positive / Negative Score ──
    ax = axes_flat[2]
    s_pos = [r.get("pos_score_mean", 0) for r in records]
    s_neg = [r.get("neg_score_mean", 0) for r in records]
    ax.plot(steps, s_pos, color="#3498db", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(s_pos), color="#3498db", linewidth=1.5, label="positive")
    ax.plot(steps, s_neg, color="#e67e22", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(s_neg), color="#e67e22", linewidth=1.5, label="negative")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Score (logit_yes - logit_no)")
    ax.set_xlabel("step")
    ax.set_ylabel("raw logit diff")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Learning Rate ──
    ax = axes_flat[3]
    lrs = [r.get("lr") for r in records]
    if any(v is not None for v in lrs):
        ax.plot(steps, lrs, color="#9b59b6", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("step")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"已保存：{output_path}")
    plt.close()


def main():
    repo_root = Path(__file__).parent.parent.parent
    default_dir = repo_root / "train/logs/qwen_sft"

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=str(default_dir))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    f0 = log_dir / "train_log_rank0.jsonl"
    f1 = log_dir / "train_log_rank1.jsonl"

    if not f0.exists() and not f1.exists():
        print(f"未找到日志文件：{f0} / {f1}")
        return

    plots = []
    if f0.exists():
        plots.append((load_log(f0), log_dir / "training_curves_rank0.png", "SFT Training Curves (Rank 0)"))
    if f1.exists():
        plots.append((load_log(f1), log_dir / "training_curves_rank1.png", "SFT Training Curves (Rank 1)"))
    if f0.exists() and f1.exists():
        mean_records = compute_mean(plots[0][0], plots[1][0])
        if mean_records:
            plots.append((mean_records, log_dir / "training_curves_mean.png", "SFT Training Curves (Mean)"))

    for records, out_path, title in plots:
        if not records:
            print(f"日志为空，跳过：{out_path.name}")
            continue
        print(f"读取 {len(records)} 条记录，最终 step={records[-1]['step']}")
        plot(records, out_path, title)


if __name__ == "__main__":
    main()
