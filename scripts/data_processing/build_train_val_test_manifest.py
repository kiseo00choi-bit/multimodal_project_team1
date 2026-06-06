from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def split_training_group(group: pd.DataFrame, val_ratio: float, rng: random.Random) -> pd.DataFrame:
    indices = list(group.index)
    rng.shuffle(indices)
    val_count = max(1, round(len(indices) * val_ratio))
    val_indices = set(indices[:val_count])
    group = group.copy()
    group["split"] = ["val" if idx in val_indices else "train" for idx in group.index]
    return group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/frames_224_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/frames_224_trainvaltest.csv"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"class_code": str})
    df = df.rename(columns={"split": "original_split"})

    training = df[df["original_split"] == "Training"].copy()
    validation = df[df["original_split"] == "Validation"].copy()

    rng = random.Random(args.seed)
    train_val_groups = []
    for _, group in training.groupby("class_id", sort=True):
        train_val_groups.append(split_training_group(group, args.val_ratio, rng))
    train_val = pd.concat(train_val_groups, ignore_index=True)
    validation["split"] = "test"

    result = pd.concat([train_val, validation], ignore_index=True)
    first_cols = ["split", "original_split"]
    other_cols = [col for col in result.columns if col not in first_cols]
    result = result[first_cols + other_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")

    print(f"wrote={args.output}")
    print(result.groupby(["split", "class_code", "label_en"]).size().to_string())


if __name__ == "__main__":
    main()
