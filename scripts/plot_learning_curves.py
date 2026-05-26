from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")
METRIC_COLUMNS = [
    ("loss", "Loss", "train_loss", "valid_loss"),
    ("macro_f1", "Macro F1", "train_macro_f1", "valid_macro_f1"),
    ("accuracy", "Accuracy", "train_accuracy", "valid_accuracy"),
]


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


def load_histories(history_paths: list[Path]) -> dict[str, pd.DataFrame]:
    histories = {}
    for history_path in history_paths:
        run_name = history_path.name.replace("_history.csv", "")
        histories[run_name] = pd.read_csv(history_path)
    return histories


def axis_limits(histories: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    max_epoch = max(int(history["epoch"].max()) for history in histories.values())
    limits = {"epoch": (1, max_epoch)}

    for metric_key, _, train_col, valid_col in METRIC_COLUMNS:
        values = []
        for history in histories.values():
            values.extend(history[train_col].tolist())
            values.extend(history[valid_col].tolist())
        ymin = min(values)
        ymax = max(values)
        if metric_key in {"macro_f1", "accuracy"}:
            limits[metric_key] = (0.0, 1.0)
        else:
            padding = max((ymax - ymin) * 0.05, 0.05)
            limits[metric_key] = (max(0.0, ymin - padding), ymax + padding)
    return limits


def plot_shared_axis_grid(histories: dict[str, pd.DataFrame], out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_names = list(histories.keys())
    limits = axis_limits(histories)

    fig, axes = plt.subplots(
        len(run_names),
        len(METRIC_COLUMNS),
        figsize=(15, 3.2 * len(run_names)),
        sharex=True,
    )
    if len(run_names) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, run_name in enumerate(run_names):
        history = histories[run_name]
        for col_idx, (metric_key, title, train_col, valid_col) in enumerate(METRIC_COLUMNS):
            ax = axes[row_idx][col_idx]
            ax.plot(history["epoch"], history[train_col], label="train", linewidth=1.8)
            ax.plot(history["epoch"], history[valid_col], label="val", linewidth=1.8)
            ax.set_xlim(*limits["epoch"])
            ax.set_ylim(*limits[metric_key])
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(run_name)
            if row_idx == len(run_names) - 1:
                ax.set_xlabel("Epoch")
            if row_idx == 0 and col_idx == len(METRIC_COLUMNS) - 1:
                ax.legend(loc="lower right")

    fig.suptitle("Learning Curves with Shared Axes")
    fig.tight_layout()
    out_path = out_dir / "all_models_learning_curves_shared_axes.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_validation_overlay(histories: dict[str, pd.DataFrame], out_dir: Path = FIGURES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = axis_limits(histories)
    written = []

    for metric_key, title, _, valid_col in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for run_name, history in histories.items():
            ax.plot(history["epoch"], history[valid_col], label=run_name, linewidth=1.8)
        ax.set_title(f"Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_xlim(*limits["epoch"])
        ax.set_ylim(*limits[metric_key])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"all_models_validation_{metric_key}_overlay.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    history_paths = sorted(args.metrics_dir.glob("*_history.csv"))
    if not args.include_smoke:
        history_paths = [path for path in history_paths if "_smoke_" not in path.name]
    if not history_paths:
        print(f"No history files found in {args.metrics_dir}")
        return

    for history_path in history_paths:
        out_path = plot_history(history_path, args.out_dir)
        print(f"wrote={out_path}")

    histories = load_histories(history_paths)
    shared_path = plot_shared_axis_grid(histories, args.out_dir)
    print(f"wrote={shared_path}")
    for out_path in plot_validation_overlay(histories, args.out_dir):
        print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
