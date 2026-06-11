#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase E：训练/验证/测试划分

策略：
- val：固定为 judge/tasks.txt 中的 6 部旧剧
- 其余 100 部按 TEST_RATIO 随机切 train/test（按剧切，避免 query 泄漏）

输出：judge/training_data/{dataset_version}/splits/
    train_pairs.jsonl / val_pairs.jsonl / test_pairs.jsonl
    train_groups.jsonl / val_groups.jsonl / test_groups.jsonl
    train_samples.jsonl / val_samples.jsonl / test_samples.jsonl
    split_manifest.json
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

LOGGER = logging.getLogger(__name__)

# val 固定剧集
VAL_DRAMAS_FILE = Path(__file__).resolve().parent / "tasks.txt"

TEST_RATIO = 0.1   # 剩余剧中 10% 作为 test
RNG_SEED = 42


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_val_dramas(tasks_file: Path) -> List[str]:
    return [l.strip() for l in tasks_file.read_text().splitlines() if l.strip()]


def find_all_dramas(v3_dir: Path) -> List[Tuple[str, Path]]:
    """返回 [(dj, run_dir), ...]"""
    result = []
    for stats_path in sorted(v3_dir.glob("*/*/stats.json")):
        dj = stats_path.parent.parent.name
        result.append((dj, stats_path.parent))
    return result


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def build_split(
    dataset_version: str = "v3",
    test_ratio: float = TEST_RATIO,
    rng_seed: int = RNG_SEED,
) -> None:
    v3_dir = repo_root() / "judge" / "training_data" / dataset_version
    out_dir = v3_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dramas = load_val_dramas(VAL_DRAMAS_FILE)
    LOGGER.info("val 固定剧集 (%d): %s", len(val_dramas), val_dramas)

    all_dramas = find_all_dramas(v3_dir)
    LOGGER.info("v3 目录共找到 %d 个剧", len(all_dramas))

    # 按 split 分组 run_dir
    val_runs, other_runs = [], []
    for dj, run_dir in all_dramas:
        if dj in val_dramas:
            val_runs.append((dj, run_dir))
        else:
            other_runs.append((dj, run_dir))

    # 按剧名去重（每个 dj 理论上只有一个 run）
    rng = random.Random(rng_seed)
    rng.shuffle(other_runs)
    n_test = max(1, round(len(other_runs) * test_ratio))
    test_runs = other_runs[:n_test]
    train_runs = other_runs[n_test:]

    LOGGER.info("train=%d  val=%d  test=%d", len(train_runs), len(val_runs), len(test_runs))

    split_map = {
        "train": train_runs,
        "val": val_runs,
        "test": test_runs,
    }

    # 合并并写入
    manifest: Dict = {"split_counts": {}, "dramas": {}}
    for split, runs in split_map.items():
        pairs, groups, samples = [], [], []
        for dj, run_dir in runs:
            pairs   += read_jsonl(run_dir / "pairs.jsonl")
            groups  += read_jsonl(run_dir / "groups.jsonl")
            samples += read_jsonl(run_dir / "samples.jsonl")
            manifest["dramas"][dj] = split

        write_jsonl(out_dir / f"{split}_pairs.jsonl",   pairs)
        write_jsonl(out_dir / f"{split}_groups.jsonl",  groups)
        write_jsonl(out_dir / f"{split}_samples.jsonl", samples)

        manifest["split_counts"][split] = {
            "dramas":   len(runs),
            "pairs":    len(pairs),
            "groups":   len(groups),
            "samples":  len(samples),
        }
        LOGGER.info(
            "%s: dramas=%d pairs=%d groups=%d samples=%d",
            split, len(runs), len(pairs), len(groups), len(samples),
        )

    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("split_manifest.json 已写入 %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E：构建 train/val/test 切分")
    parser.add_argument("--dataset-version", default="v3")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    build_split(args.dataset_version, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
