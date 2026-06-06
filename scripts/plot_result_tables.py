from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("docs/assets/result_tables")

BG = "#071821"
HEADER = "#0b4d56"
ROW_A = "#0d2530"
ROW_B = "#102f3a"
EDGE = "#19d9d2"
TEXT = "#f2fbff"
MUTED = "#b8cbd3"
BEST = "#143f3a"


EXPERIMENT1_MODELS = [
    ("baseline_cnn_avg", "CNN + Average Pooling"),
    ("baseline_cnn_lstm", "CNN + GRU"),
    ("baseline_keypoint", "GT Keypoint 1D-CNN + GRU"),
    ("fusion", "RGB + GT Keypoint Fusion"),
    ("fusion_attention", "RGB + GT Keypoint Cross-Attention"),
]

EXPERIMENT2_MODELS = {
    "rgb_only": "RGB Only",
    "pred_keypoint_only": "RGB -> Predicted Keypoint Only",
    "pred_keypoint_fusion": "RGB + Predicted Keypoint Fusion",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def experiment1_rows() -> list[list[str]]:
    metrics_dir = Path("outputs/experiment1/metrics")
    rows = []
    for run_name, label in EXPERIMENT1_MODELS:
        summary = load_json(metrics_dir / f"{run_name}_summary.json")
        test_eval = load_json(metrics_dir / f"{run_name}_test_best_eval.json")
        rows.append(
            [
                label,
                fmt(int(test_eval["checkpoint_epoch"])),
                fmt(float(summary["best_valid_macro_f1"])),
                fmt(float(test_eval["accuracy"])),
                fmt(float(test_eval["macro_f1"])),
            ]
        )
    return rows


def experiment2_rows() -> list[list[str]]:
    summary = load_json(Path("outputs/experiment2/metrics/experiment2_summary.json"))
    rows = []
    for item in summary["classifiers"]:
        rows.append(
            [
                EXPERIMENT2_MODELS[item["mode"]],
                fmt(int(item["best_epoch"])),
                fmt(float(item["best_valid_macro_f1"])),
                fmt(float(item["test_accuracy"])),
                fmt(float(item["test_macro_f1"])),
            ]
        )
    return rows


def draw_table(title: str, subtitle: str, rows: list[list[str]], path: Path) -> None:
    columns = ["Model", "Best Epoch", "Valid Macro F1", "Test Accuracy", "Test Macro F1"]
    fig_h = 2.2 + len(rows) * 0.48
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(0.02, 0.94, title, color="#43fff6", fontsize=22, fontweight="bold", ha="left", transform=ax.transAxes)
    ax.text(0.02, 0.865, subtitle, color=MUTED, fontsize=11.5, ha="left", transform=ax.transAxes)

    best_macro = max(float(row[-1]) for row in rows)
    cell_colours = []
    for idx, row in enumerate(rows):
        base = ROW_A if idx % 2 == 0 else ROW_B
        colors = [base] * len(columns)
        if abs(float(row[-1]) - best_macro) < 1e-9:
            colors[-1] = BEST
        cell_colours.append(colors)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellColours=cell_colours,
        colColours=[HEADER] * len(columns),
        colWidths=[0.42, 0.13, 0.15, 0.15, 0.15],
        cellLoc="center",
        bbox=[0.02, 0.08, 0.96, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.8)
        cell.get_text().set_color(TEXT)
        if row == 0:
            cell.get_text().set_weight("bold")
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")

    ax.text(
        0.02,
        0.02,
        "Best model is highlighted by Test Macro F1.",
        color=MUTED,
        fontsize=10.5,
        ha="left",
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_table(
        "Experiment 1 Results: GT Keypoint Baselines",
        "Validation selects the best checkpoint; test metrics are measured on the AI Hub Validation hold-out split.",
        experiment1_rows(),
        OUT_DIR / "experiment1_results_table.png",
    )
    draw_table(
        "Experiment 2 Results: Predicted Keypoint Downstream",
        "GT keypoints supervise only the pose estimator; downstream classifiers use RGB and/or predicted keypoints.",
        experiment2_rows(),
        OUT_DIR / "experiment2_results_table.png",
    )
    print(OUT_DIR / "experiment1_results_table.png")
    print(OUT_DIR / "experiment2_results_table.png")


if __name__ == "__main__":
    main()
