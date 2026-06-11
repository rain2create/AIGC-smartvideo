"""
plot_awr_training.py  —  AWR v2 训练曲线图

从 train_log_rank{0,1}.jsonl 生成训练曲线。

9 panels (3x3):
  1. Total Loss
  2. PG Loss vs Anchor Loss
  3. Delta (policy - ref) Mean & Std
  4. Advantage Mean
  5. Policy Score Mean / Ref Score Mean
  6. Score Raw Max (explosion monitor)
  7. Norm Weight (group weighting)
  8. Reward Mean & Std
  9. Learning Rate

用法：
  python train/listwise_awr/plot_awr_training.py
  python train/listwise_awr/plot_awr_training.py --log-dir train/logs/qwen_awr_v2
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
    "loss", "pg_loss", "anchor_loss",
    "delta_mean", "delta_std",
    "advantage_mean",
    "score_policy_mean", "score_ref_mean",
    "reward_mean", "reward_std",
    "norm_weight", "avg_group_size",
]
_MAX_KEYS = ["score_raw_max"]
_OPTIONAL_KEYS = ["ratio_mean", "clip_frac"]


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
        for k in _OPTIONAL_KEYS:
            if k in a and k in b:
                rec[k] = (a[k] + b[k]) / 2
        mean.append(rec)
    return mean


def _smooth(values, window=10):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return list(values[: window - 1]) + list(smoothed)


def plot(records, output_path: Path, title: str):
    steps = [r["step"] for r in records]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax = axes.flatten()
    p = 0

    # ── 0: Total Loss ──
    a = ax[p]; p += 1
    vals = [r["loss"] for r in records]
    a.plot(steps, vals, color="#e74c3c", alpha=0.3, linewidth=0.8)
    a.plot(steps, _smooth(vals), color="#e74c3c", linewidth=1.5, label="smoothed")
    a.set_title("Total Loss")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── 1: PG Loss vs Anchor Loss ──
    a = ax[p]; p += 1
    pg = [r.get("pg_loss", 0) for r in records]
    anch = [r.get("anchor_loss", 0) for r in records]
    a.plot(steps, pg, color="#3498db", alpha=0.2, linewidth=0.8)
    a.plot(steps, _smooth(pg), color="#3498db", linewidth=1.5, label="pg_loss")
    a.plot(steps, anch, color="#e67e22", alpha=0.2, linewidth=0.8)
    a.plot(steps, _smooth(anch), color="#e67e22", linewidth=1.5, label="anchor_loss")
    a.set_title("PG Loss vs Anchor Loss")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── 2: Delta Mean & Std ──
    a = ax[p]; p += 1
    dm = [r.get("delta_mean", 0) for r in records]
    ds = [r.get("delta_std", 0) for r in records]
    a.plot(steps, _smooth(dm), color="#3498db", linewidth=1.5, label="δ mean")
    a.fill_between(steps,
                    [m - s for m, s in zip(_smooth(dm), _smooth(ds))],
                    [m + s for m, s in zip(_smooth(dm), _smooth(ds))],
                    alpha=0.15, color="#3498db")
    a.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    a.set_title("Delta (policy - ref)")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── 3: Advantage Mean ──
    a = ax[p]; p += 1
    adv = [r.get("advantage_mean", 0) for r in records]
    a.plot(steps, adv, color="#8e44ad", alpha=0.2, linewidth=0.8)
    a.plot(steps, _smooth(adv), color="#8e44ad", linewidth=1.5)
    a.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    a.set_title("Advantage Mean (z-scored)")
    a.set_xlabel("step")
    a.grid(True, alpha=0.3)

    # ── 4: Policy / Ref Score ──
    a = ax[p]; p += 1
    sp = [r.get("score_policy_mean", 0) for r in records]
    sr = [r.get("score_ref_mean", 0) for r in records]
    a.plot(steps, _smooth(sp), color="#3498db", linewidth=1.5, label="s_policy")
    a.plot(steps, _smooth(sr), color="#e67e22", linewidth=1.5, label="s_ref")
    a.set_title("Policy vs Reference Score")
    a.set_xlabel("step")
    a.set_ylabel("raw logit diff")
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── 5: Score Raw Max ──
    a = ax[p]; p += 1
    rm = [r.get("score_raw_max", 0) for r in records]
    a.plot(steps, rm, color="#c0392b", alpha=0.3, linewidth=0.8)
    a.plot(steps, _smooth(rm), color="#c0392b", linewidth=1.5)
    a.axhline(y=15, color="orange", linestyle="--", linewidth=0.8, label="warning=15")
    a.set_title("Score Raw Max (explosion monitor)")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── 6: Norm Weight ──
    a = ax[p]; p += 1
    nw = [r.get("norm_weight", 1.0) for r in records]
    a.plot(steps, nw, color="#27ae60", alpha=0.3, linewidth=0.8)
    a.plot(steps, _smooth(nw), color="#27ae60", linewidth=1.5)
    a.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    a.set_title("Norm Weight (group weighting)")
    a.set_xlabel("step")
    a.grid(True, alpha=0.3)

    # ── 7: Reward Mean & Std ──
    a = ax[p]; p += 1
    rmean = [r.get("reward_mean", 0) for r in records]
    rstd = [r.get("reward_std", 0) for r in records]
    a.plot(steps, _smooth(rmean), color="#2ecc71", linewidth=1.5, label="reward mean")
    a.plot(steps, _smooth(rstd), color="#e74c3c", linewidth=1.5, label="reward std")
    a.set_title("Reward Distribution")
    a.set_xlabel("step")
    a.legend(fontsize=8)
    a.grid(True, alpha=0.3)

    # ── 8: Learning Rate ──
    a = ax[p]; p += 1
    lrs = [r.get("lr") for r in records]
    if any(v is not None for v in lrs):
        a.plot(steps, lrs, color="#9b59b6", linewidth=1.5)
    a.set_title("Learning Rate")
    a.set_xlabel("step")
    a.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"已保存：{output_path}")
    plt.close()


def main():
    repo_root = Path(__file__).parent.parent.parent
    default_dir = repo_root / "train/logs/qwen_awr_v2"

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
                       "AWR v2 Training Curves (Rank 0)"))
    if f1.exists():
        plots.append((load_log(f1), log_dir / "training_curves_rank1.png",
                       "AWR v2 Training Curves (Rank 1)"))
    if f0.exists() and f1.exists():
        mean_records = compute_mean(plots[0][0], plots[1][0])
        if mean_records:
            plots.append((mean_records, log_dir / "training_curves_mean.png",
                           "AWR v2 Training Curves (Mean)"))

    for records, out_path, title in plots:
        if not records:
            print(f"日志为空，跳过：{out_path.name}")
            continue
        print(f"读取 {len(records)} 条记录，最终 step={records[-1]['step']}")
        plot(records, out_path, title)


if __name__ == "__main__":
    main()
