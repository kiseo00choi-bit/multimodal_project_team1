from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")


def plot_history(history_path: Path, out_dir: Path = FIGURES_DIR) -> Path:
    history = pd.read_csv(history_path)
    run_name = history_path.name.replace("_history.csv", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["valid_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_macro_f1"], label="train")
    axes[1].plot(history["epoch"], history["valid_macro_f1"], label="val")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[2].plot(history["epoch"], history["valid_accuracy"], label="val")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle(run_name)
    fig.tight_layout()
    out_path = out_dir / f"{run_name}_learning_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    history_paths = sorted(args.metrics_dir.glob("*_history.csv"))
    if not history_paths:
        print(f"No history files found in {args.metrics_dir}")
        return

    for history_path in history_paths:
        out_path = plot_history(history_path, args.out_dir)
        print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
