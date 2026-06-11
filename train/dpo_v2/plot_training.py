"""
plot_training.py  —  从 train_log_rank{0,1}.jsonl 生成 DPO v2 训练曲线图

生成三张图：
  training_curves_rank0.png   — rank 0 本卡指标
  training_curves_rank1.png   — rank 1 本卡指标
  training_curves_mean.png    — 两卡按 step 对齐后的均值

9 panels (3x3):
  1. Total Loss
  2. Preference Loss vs Anchor Loss
  3. Policy Scores (s_w / s_r)
  4. Reward Margin
  5. DPO Advantage (δw - δr)
  6. Advantage Win Rate
  7. Policy - Reference Deltas
  8. Score Raw Max (explosion monitor)
  9. Learning Rate

用法：
  python train/dpo_v2/plot_training.py
  python train/dpo_v2/plot_training.py --log-dir train/logs/qwen_dpo_v2
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


# 需要取均值的字段
_MEAN_KEYS = [
    "loss", "s_w_mean", "s_r_mean", "reward_margin",
    "preference_loss", "anchor_loss",
    "delta_w", "delta_r", "adv_mean", "adv_pos_rate",
]
# 取 max 的字段
_MAX_KEYS = ["score_raw_max"]


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
        for k in _MAX_KEYS:
            rec[k] = max(a.get(k, 0), b.get(k, 0))
        mean.append(rec)
    return mean


def _smooth(values, window=20):
    """简单移动平均平滑。"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # 前 window-1 个点保持原始值
    return list(values[: window - 1]) + list(smoothed)


def plot(records, output_path: Path, title: str):
    steps = [r["step"] for r in records]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    panel = 0

    # ── Panel 0: Total Loss ──
    ax = axes_flat[panel]; panel += 1
    losses = [r["loss"] for r in records]
    ax.plot(steps, losses, color="#e74c3c", alpha=0.3, linewidth=0.8)
    ax.plot(steps, _smooth(losses), color="#e74c3c", linewidth=1.5, label="smoothed")
    ax.axhline(y=0.6931, color="gray", linestyle="--", linewidth=0.8, label="log(2)")
    ax.set_title("Total Loss")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 1: Preference Loss vs Anchor Loss ──
    ax = axes_flat[panel]; panel += 1
    pref = [r.get("preference_loss", 0) for r in records]
    anch = [r.get("anchor_loss", 0) for r in records]
    ax.plot(steps, pref, color="#3498db", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(pref), color="#3498db", linewidth=1.5, label="preference")
    ax.plot(steps, anch, color="#e67e22", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(anch), color="#e67e22", linewidth=1.5, label="anchor")
    ax.axhline(y=0.6931, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Preference Loss vs Anchor Loss")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: s_w / s_r (Policy Scores) ──
    ax = axes_flat[panel]; panel += 1
    s_w = [r.get("s_w_mean", 0) for r in records]
    s_r = [r.get("s_r_mean", 0) for r in records]
    ax.plot(steps, s_w, color="#3498db", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(s_w), color="#3498db", linewidth=1.5, label="s_w (chosen)")
    ax.plot(steps, s_r, color="#e67e22", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(s_r), color="#e67e22", linewidth=1.5, label="s_r (rejected)")
    ax.set_title("Policy Scores")
    ax.set_xlabel("step")
    ax.set_ylabel("raw logit diff")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Reward Margin ──
    ax = axes_flat[panel]; panel += 1
    margins = [r.get("reward_margin", 0) for r in records]
    ax.plot(steps, margins, color="#2ecc71", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(margins), color="#2ecc71", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Reward Margin (s_w - s_r)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: adv_mean ──
    ax = axes_flat[panel]; panel += 1
    adv = [r.get("adv_mean", 0) for r in records]
    ax.plot(steps, adv, color="#8e44ad", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(adv), color="#8e44ad", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("DPO Advantage  (δw - δr)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    # ── Panel 5: adv_pos_rate ──
    ax = axes_flat[panel]; panel += 1
    adv_pos = [r.get("adv_pos_rate", 0) for r in records]
    ax.plot(steps, adv_pos, color="#27ae60", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(adv_pos), color="#27ae60", linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
    ax.set_title("Advantage Win Rate")
    ax.set_xlabel("step")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: delta_w / delta_r ──
    ax = axes_flat[panel]; panel += 1
    dw = [r.get("delta_w", 0) for r in records]
    dr = [r.get("delta_r", 0) for r in records]
    ax.plot(steps, dw, color="#3498db", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(dw), color="#3498db", linewidth=1.5, label="δw (chosen)")
    ax.plot(steps, dr, color="#e67e22", alpha=0.2, linewidth=0.8)
    ax.plot(steps, _smooth(dr), color="#e67e22", linewidth=1.5, label="δr (rejected)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Policy - Reference Deltas")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 7: score_raw_max ──
    ax = axes_flat[panel]; panel += 1
    raw_max = [r.get("score_raw_max", 0) for r in records]
    ax.plot(steps, raw_max, color="#c0392b", alpha=0.3, linewidth=0.8)
    ax.plot(steps, _smooth(raw_max), color="#c0392b", linewidth=1.5)
    ax.set_title("Score Raw Max (explosion monitor)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    # ── Panel 8: Learning Rate ──
    ax = axes_flat[panel]; panel += 1
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
    default_dir = repo_root / "train/logs/qwen_dpo_v2"

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
        plots.append((load_log(f0), log_dir / "training_curves_rank0.png",
                       "DPO v2 Training Curves (Rank 0)"))
    if f1.exists():
        plots.append((load_log(f1), log_dir / "training_curves_rank1.png",
                       "DPO v2 Training Curves (Rank 1)"))
    if f0.exists() and f1.exists():
        mean_records = compute_mean(plots[0][0], plots[1][0])
        if mean_records:
            plots.append((mean_records, log_dir / "training_curves_mean.png",
                           "DPO v2 Training Curves (Mean)"))

    for records, out_path, title in plots:
        if not records:
            print(f"日志为空，跳过：{out_path.name}")
            continue
        print(f"读取 {len(records)} 条记录，最终 step={records[-1]['step']}")
        plot(records, out_path, title)


if __name__ == "__main__":
    main()
