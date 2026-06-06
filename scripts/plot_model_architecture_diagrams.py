from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("docs/assets/model_architectures")

BG = "#071821"
CARD = "#0b2029"
BLOCK = "#112b35"
EDGE = "#19d9d2"
TEXT = "#f2fbff"
MUTED = "#b8cbd3"
RGB_COLOR = "#173f66"
GT_KEYPOINT_COLOR = "#6b2f4a"
PRED_KEYPOINT_COLOR = "#462d64"
OUTPUT_COLOR = "#0e8f86"


def add_round_rect(ax, x, y, w, h, label, fc=BLOCK, ec=EDGE, fontsize=13, lw=1.8):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.022",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=fontsize,
        fontweight="medium",
        linespacing=1.18,
    )
    return patch


def add_arrow(ax, x0, y0, x1, y1, color=MUTED, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            shrinkA=5,
            shrinkB=5,
        )
    )


def setup(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.965, title, color="#43fff6", fontsize=28, fontweight="bold", ha="left")
    ax.text(0.04, 0.928, subtitle, color=MUTED, fontsize=15, ha="left")
    return fig, ax


def draw_pipeline_card(ax, x, y, w, h, title, inputs, steps, output="8 classes", title_color="#1b4d5f"):
    add_round_rect(ax, x, y, w, h, "", fc=CARD, ec=EDGE, lw=1.6)
    ax.text(x + 0.025, y + h - 0.035, title, color="#43fff6", fontsize=17, fontweight="bold", ha="left")

    input_text = "\n".join(inputs)
    block_h = min(0.075, max(0.062, h * 0.38))
    block_y = y + 0.032
    add_round_rect(ax, x + 0.025, block_y, 0.18 * w, block_h, input_text, fc=title_color, fontsize=12)

    step_count = len(steps)
    gap = 0.018 * w
    start_x = x + 0.24 * w
    end_x = x + w - 0.12 * w
    step_w = (end_x - start_x - gap * (step_count - 1)) / step_count
    step_h = block_h
    step_y = block_y

    prev_right = x + 0.025 + 0.18 * w
    mid_y = step_y + step_h / 2
    for idx, step in enumerate(steps):
        sx = start_x + idx * (step_w + gap)
        add_arrow(ax, prev_right, mid_y, sx, mid_y)
        add_round_rect(ax, sx, step_y, step_w, step_h, step, fc=BLOCK, fontsize=12)
        prev_right = sx + step_w

    out_x = x + w - 0.095 * w
    add_arrow(ax, prev_right, mid_y, out_x, mid_y)
    add_round_rect(ax, out_x, step_y, 0.07 * w, step_h, output, fc=OUTPUT_COLOR, fontsize=12)


def draw_fusion_card(
    ax,
    x,
    y,
    w,
    h,
    title,
    rgb_input_label,
    pose_input_label,
    rgb_label,
    pose_label,
    fusion_label,
    rgb_color=RGB_COLOR,
    pose_color=PRED_KEYPOINT_COLOR,
):
    add_round_rect(ax, x, y, w, h, "", fc=CARD, ec=EDGE, lw=1.6)
    ax.text(x + 0.025, y + h - 0.038, title, color="#43fff6", fontsize=17, fontweight="bold", ha="left")

    input_x = x + 0.025
    input_w = 0.18 * w
    input_h = h * 0.20
    rgb_input_y = y + h * 0.40
    pose_input_y = y + h * 0.16
    add_round_rect(ax, input_x, rgb_input_y, input_w, input_h, rgb_input_label, fc=rgb_color, fontsize=11.5)
    add_round_rect(ax, input_x, pose_input_y, input_w, input_h, pose_input_label, fc=pose_color, fontsize=11.5)

    branch_w = 0.18 * w
    branch_h = h * 0.20
    branch_x = x + 0.25 * w
    rgb_y = rgb_input_y
    pose_y = pose_input_y
    add_round_rect(ax, branch_x, rgb_y, branch_w, branch_h, rgb_label, fc=BLOCK, fontsize=11.5)
    add_round_rect(ax, branch_x, pose_y, branch_w, branch_h, pose_label, fc=BLOCK, fontsize=11.5)

    fusion_x = x + 0.49 * w
    fusion_y = y + h * 0.255
    fusion_w = 0.15 * w
    fusion_h = h * 0.27
    add_round_rect(ax, fusion_x, fusion_y, fusion_w, fusion_h, fusion_label, fc=BLOCK, fontsize=12)

    cls_x = x + 0.69 * w
    cls_w = 0.17 * w
    add_round_rect(ax, cls_x, fusion_y, cls_w, fusion_h, "Classifier", fc=BLOCK, fontsize=12)

    out_x = x + 0.88 * w
    out_w = 0.07 * w
    add_round_rect(ax, out_x, fusion_y, out_w, fusion_h, "8 classes", fc=OUTPUT_COLOR, fontsize=12)

    add_arrow(ax, input_x + input_w, rgb_input_y + input_h / 2, branch_x, rgb_y + branch_h / 2)
    add_arrow(ax, input_x + input_w, pose_input_y + input_h / 2, branch_x, pose_y + branch_h / 2)
    add_arrow(ax, branch_x + branch_w, rgb_y + branch_h / 2, fusion_x, fusion_y + fusion_h * 0.65)
    add_arrow(ax, branch_x + branch_w, pose_y + branch_h / 2, fusion_x, fusion_y + fusion_h * 0.35)
    add_arrow(ax, fusion_x + fusion_w, fusion_y + fusion_h / 2, cls_x, fusion_y + fusion_h / 2)
    add_arrow(ax, cls_x + cls_w, fusion_y + fusion_h / 2, out_x, fusion_y + fusion_h / 2)


def draw_experiment1():
    fig, ax = setup(
        "Experiment 1: GT Keypoint Baselines",
        "GT keypoints from XML are directly used as classifier inputs.",
    )

    draw_pipeline_card(
        ax,
        0.04,
        0.71,
        0.92,
        0.16,
        "RGB CNN + Average Pooling",
        ["RGB frames", "16 sampled\nframes"],
        ["ResNet18\nImageNet features", "Average\nPooling", "Linear\nClassifier"],
        title_color="#0d4550",
    )
    draw_pipeline_card(
        ax,
        0.04,
        0.51,
        0.92,
        0.16,
        "RGB CNN + GRU",
        ["RGB frames", "16 sampled\nframes"],
        ["ResNet18\nImageNet features", "GRU temporal\nmodel", "Linear\nClassifier"],
        title_color="#173f66",
    )
    draw_pipeline_card(
        ax,
        0.04,
        0.31,
        0.92,
        0.16,
        "GT Keypoint 1D-CNN + GRU",
        ["XML GT\nkeypoints", "[T, 34]"],
        ["1D-CNN\npose encoder", "GRU temporal\nmodel", "Linear\nClassifier"],
        title_color=GT_KEYPOINT_COLOR,
    )
    draw_fusion_card(
        ax,
        0.04,
        0.06,
        0.92,
        0.21,
        "RGB + GT Keypoint Fusion",
        "RGB frames",
        "XML GT\nkeypoints",
        "RGB branch\nResNet18 + GRU",
        "Pose branch\n1D-CNN + GRU",
        "Concat /\nAttention",
        pose_color=GT_KEYPOINT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "experiment1_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_stage1_card(ax):
    add_round_rect(ax, 0.04, 0.68, 0.92, 0.22, "", fc=CARD, ec=EDGE, lw=1.6)
    ax.text(0.065, 0.855, "Stage 1: Image -> Keypoint Estimator", color="#43fff6", fontsize=18, fontweight="bold")
    ax.text(
        0.065,
        0.818,
        "GT keypoints supervise this stage only. Downstream classifiers never receive GT keypoints.",
        color=MUTED,
        fontsize=13,
    )

    blocks = [
        ("RGB frames\n16 sampled frames", "#0d4550"),
        ("ResNet18\nframe encoder", BLOCK),
        ("Keypoint head\nxy + visibility", BLOCK),
        ("Predicted keypoints\n[T, 34]", PRED_KEYPOINT_COLOR),
    ]
    x = 0.08
    y = 0.745
    w = 0.17
    h = 0.055
    for idx, (label, color) in enumerate(blocks):
        add_round_rect(ax, x, y, w, h, label, fc=color, fontsize=11.5)
        if idx < len(blocks) - 1:
            add_arrow(ax, x + w, y + h / 2, x + w + 0.045, y + h / 2)
        x += w + 0.065
    add_round_rect(ax, 0.55, 0.686, 0.17, 0.040, "XML GT keypoints\ntraining target", fc=GT_KEYPOINT_COLOR, fontsize=10.5)
    add_arrow(ax, 0.635, 0.726, 0.635, 0.745, color="#d5c5ff", linestyle="--")


def draw_experiment2():
    fig, ax = setup(
        "Experiment 2: Predicted Keypoint Downstream",
        "The pipeline is rebuilt for practical inference: RGB-only vs predicted-keypoint-only vs fusion.",
    )

    draw_stage1_card(ax)

    draw_pipeline_card(
        ax,
        0.04,
        0.51,
        0.92,
        0.15,
        "RGB Only",
        ["RGB frames", "16 sampled\nframes"],
        ["ResNet18\nfeatures", "GRU temporal\nmodel", "Classifier"],
        title_color=RGB_COLOR,
    )
    draw_pipeline_card(
        ax,
        0.04,
        0.32,
        0.92,
        0.15,
        "RGB -> Predicted Keypoint Only",
        ["Predicted\nkeypoints", "[T, 34]"],
        ["1D-CNN\npose encoder", "GRU temporal\nmodel", "Classifier"],
        title_color=PRED_KEYPOINT_COLOR,
    )
    draw_fusion_card(
        ax,
        0.04,
        0.07,
        0.92,
        0.20,
        "RGB + Predicted Keypoint Fusion",
        "RGB frames",
        "Predicted\nkeypoints",
        "RGB branch\nResNet18 + GRU",
        "Pose branch\n1D-CNN + GRU",
        "Concat\nFusion",
        pose_color=PRED_KEYPOINT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "experiment2_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_experiment1()
    draw_experiment2()
    print(OUT_DIR / "experiment1_architecture.png")
    print(OUT_DIR / "experiment2_architecture.png")


if __name__ == "__main__":
    main()
