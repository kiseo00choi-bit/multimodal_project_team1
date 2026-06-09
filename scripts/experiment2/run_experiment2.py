from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import AbnormalBehaviorDataset
from src.models.experiment2 import (
    ImageKeypointEstimator,
    KeypointSequenceClassifier,
    RGBPredictedKeypointFusionClassifier,
    RGBSequenceClassifier,
)
from src.train import classification_metrics, confusion_matrix_np
from src.utils import device_name, set_seed


CLASS_NAMES = [
    "fall",
    "broken",
    "fire",
    "smoke",
    "abandon",
    "theft",
    "fight",
    "weak_pedestrian",
]


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_loader(args: argparse.Namespace, split: str, shuffle: bool, smoke: bool) -> DataLoader:
    limit = args.smoke_limit_per_class if smoke else args.limit_per_class
    dataset = AbnormalBehaviorDataset(
        manifest_path=args.manifest,
        split=split,
        num_frames=args.num_frames if not smoke else min(args.num_frames, 4),
        use_frames=True,
        use_keypoints=True,
        image_size=args.image_size,
        action_only=True,
        limit_per_class=limit,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if not smoke else min(args.batch_size, 2),
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def keypoint_loss_and_metrics(
    pred_xy: torch.Tensor,
    pred_visibility_logits: torch.Tensor,
    target_keypoints: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    target_xy = target_keypoints.reshape(*target_keypoints.shape[:2], 17, 2)
    visible = (target_xy.abs().sum(dim=-1) > 0).float()

    coord_error = (pred_xy - target_xy).pow(2).sum(dim=-1)
    visible_count = visible.sum().clamp_min(1.0)
    coord_loss = (coord_error * visible).sum() / visible_count
    visibility_loss = F.binary_cross_entropy_with_logits(pred_visibility_logits, visible)
    loss = coord_loss + 0.1 * visibility_loss

    mpjpe = (coord_error.sqrt() * visible).sum() / visible_count
    visibility_pred = (torch.sigmoid(pred_visibility_logits) >= 0.5).float()
    visibility_acc = (visibility_pred == visible).float().mean()
    return loss, {
        "coord_loss": float(coord_loss.detach().cpu()),
        "visibility_loss": float(visibility_loss.detach().cpu()),
        "mpjpe_norm": float(mpjpe.detach().cpu()),
        "visibility_accuracy": float(visibility_acc.detach().cpu()),
    }


def train_pose_epoch(model, loader, optimizer, device: torch.device, train: bool) -> dict:
    model.train(train)
    losses = []
    coord_losses = []
    vis_losses = []
    mpjpes = []
    vis_accs = []
    desc = "pose train" if train else "pose valid"
    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            xy, vis_logits = model.forward_raw(frames)
            loss, metrics = keypoint_loss_and_metrics(xy, vis_logits, target)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        coord_losses.append(metrics["coord_loss"])
        vis_losses.append(metrics["visibility_loss"])
        mpjpes.append(metrics["mpjpe_norm"])
        vis_accs.append(metrics["visibility_accuracy"])
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "coord_loss": float(np.mean(coord_losses)) if coord_losses else 0.0,
        "visibility_loss": float(np.mean(vis_losses)) if vis_losses else 0.0,
        "mpjpe_norm": float(np.mean(mpjpes)) if mpjpes else 0.0,
        "visibility_accuracy": float(np.mean(vis_accs)) if vis_accs else 0.0,
    }


def run_classifier_epoch(
    pose_estimator: ImageKeypointEstimator | None,
    classifier: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    mode: str,
) -> dict:
    if pose_estimator is not None:
        pose_estimator.eval()
    classifier.train(train)
    losses = []
    preds = []
    labels_all = []
    desc = f"{mode} train" if train else f"{mode} valid"
    for batch in tqdm(loader, desc=desc, leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        predicted_keypoints = None
        if mode in {"pred_keypoint_only", "pred_keypoint_fusion"}:
            if pose_estimator is None:
                raise ValueError("pose_estimator is required for predicted-keypoint modes")
            with torch.no_grad():
                predicted_keypoints = pose_estimator(frames)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            if mode == "rgb_only":
                logits = classifier(frames)
            elif mode == "pred_keypoint_only":
                logits = classifier(predicted_keypoints)
            elif mode == "pred_keypoint_fusion":
                logits = classifier(frames, predicted_keypoints)
            else:
                raise ValueError(mode)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    metrics = classification_metrics(labels_all, preds, num_classes=8)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "labels": labels_all,
        "preds": preds,
        "class_f1": metrics["f1"].tolist(),
        "class_support": metrics["support"].tolist(),
    }


def save_history_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_confusion(labels: list[int], preds: list[int], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix_np(labels, preds, num_classes=8)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(8), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(8), CLASS_NAMES)
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            ax.text(x, y, str(cm[y, x]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_learning_curve(history: list[dict], path: Path, title: str, metric_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if metric_prefix == "pose":
        axes[1].plot(epochs, [row["train_mpjpe_norm"] for row in history], label="train")
        axes[1].plot(epochs, [row["valid_mpjpe_norm"] for row in history], label="val")
        axes[1].set_title("Normalized MPJPE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[2].plot(epochs, [row["train_visibility_accuracy"] for row in history], label="train")
        axes[2].plot(epochs, [row["valid_visibility_accuracy"] for row in history], label="val")
        axes[2].set_title("Visibility Accuracy")
    else:
        axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="train")
        axes[1].plot(epochs, [row["valid_macro_f1"] for row in history], label="val")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Macro F1")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[2].plot(epochs, [row["train_accuracy"] for row in history], label="train")
        axes[2].plot(epochs, [row["valid_accuracy"] for row in history], label="val")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_pose_estimator(args, loaders: dict[str, DataLoader], device: torch.device, output_dir: Path, smoke: bool) -> dict:
    model = ImageKeypointEstimator(
        pretrained=not smoke and args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.pose_lr,
        weight_decay=args.weight_decay,
    )
    epochs = args.smoke_epochs if smoke else args.pose_epochs
    history = []
    best_mpjpe = float("inf")
    best_path = output_dir / "checkpoints" / "image_keypoint_estimator_best.pt"
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = train_pose_epoch(model, loaders["train"], optimizer, device, train=True)
        valid_metrics = train_pose_epoch(model, loaders["val"], optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mpjpe_norm": train_metrics["mpjpe_norm"],
            "train_visibility_accuracy": train_metrics["visibility_accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_mpjpe_norm": valid_metrics["mpjpe_norm"],
            "valid_visibility_accuracy": valid_metrics["visibility_accuracy"],
        }
        history.append(row)
        print(
            "pose_estimator "
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"train_mpjpe={row['train_mpjpe_norm']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_mpjpe={row['valid_mpjpe_norm']:.4f}"
        )
        if valid_metrics["mpjpe_norm"] < best_mpjpe:
            best_mpjpe = valid_metrics["mpjpe_norm"]
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "valid_mpjpe_norm": best_mpjpe,
                },
                best_path,
            )
    save_history_csv(history, output_dir / "metrics" / "image_keypoint_estimator_history.csv")
    save_learning_curve(
        history,
        output_dir / "figures" / "image_keypoint_estimator_learning_curve.png",
        "Image -> Keypoint Estimator",
        "pose",
    )
    return {
        "model": model,
        "checkpoint_path": str(best_path),
        "best_valid_mpjpe_norm": float(best_mpjpe),
        "best_epoch": int(min(history, key=lambda row: row["valid_mpjpe_norm"])["epoch"]),
        "seconds": round(time.time() - start, 2),
    }


def load_pose_checkpoint(model: ImageKeypointEstimator, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()


def train_classifier(
    args,
    pose_estimator: ImageKeypointEstimator | None,
    loaders: dict[str, DataLoader],
    device: torch.device,
    output_dir: Path,
    mode: str,
    smoke: bool,
) -> dict:
    if mode == "rgb_only":
        classifier: nn.Module = RGBSequenceClassifier(
            pretrained=not smoke and args.pretrained,
            freeze_backbone=args.freeze_backbone,
        )
    elif mode == "pred_keypoint_only":
        classifier: nn.Module = KeypointSequenceClassifier()
    elif mode == "pred_keypoint_fusion":
        classifier = RGBPredictedKeypointFusionClassifier(
            pretrained=not smoke and args.pretrained,
            freeze_backbone=args.freeze_backbone,
        )
    else:
        raise ValueError(mode)
    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (param for param in classifier.parameters() if param.requires_grad),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    epochs = args.smoke_epochs if smoke else args.classifier_epochs
    history = []
    best_f1 = -1.0
    best_path = output_dir / "checkpoints" / f"{mode}_best.pt"
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = run_classifier_epoch(
            pose_estimator, classifier, loaders["train"], criterion, optimizer, device, True, mode
        )
        valid_metrics = run_classifier_epoch(
            pose_estimator, classifier, loaders["val"], criterion, optimizer, device, False, mode
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"{mode} epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} valid_loss={row['valid_loss']:.4f} "
            f"valid_f1={row['valid_macro_f1']:.4f}"
        )
        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": classifier.state_dict(),
                    "args": vars(args),
                    "mode": mode,
                    "epoch": epoch,
                    "valid_macro_f1": best_f1,
                },
                best_path,
            )
            save_confusion(
                valid_metrics["labels"],
                valid_metrics["preds"],
                output_dir / "figures" / f"{mode}_valid_confusion.png",
                f"{mode} validation confusion",
            )

    save_history_csv(history, output_dir / "metrics" / f"{mode}_history.csv")
    save_learning_curve(
        history,
        output_dir / "figures" / f"{mode}_learning_curve.png",
        mode,
        "classifier",
    )

    checkpoint = torch.load(best_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])
    test_metrics = run_classifier_epoch(
        pose_estimator, classifier, loaders["test"], criterion, None, device, False, mode
    )
    save_confusion(
        test_metrics["labels"],
        test_metrics["preds"],
        output_dir / "figures" / f"{mode}_test_confusion.png",
        f"{mode} test confusion",
    )
    result = {
        "mode": mode,
        "checkpoint_path": str(best_path),
        "best_epoch": int(checkpoint["epoch"]),
        "best_valid_macro_f1": float(best_f1),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_class_f1": test_metrics["class_f1"],
        "test_class_support": test_metrics["class_support"],
        "seconds": round(time.time() - start, 2),
    }
    save_json(result, output_dir / "metrics" / f"{mode}_test_summary.json")
    return result


def save_comparison_figure(results: list[dict], output_dir: Path) -> None:
    names = [row["mode"].replace("pred_", "").replace("_", " ") for row in results]
    valid_f1 = [row["best_valid_macro_f1"] for row in results]
    test_f1 = [row["test_macro_f1"] for row in results]
    test_acc = [row["test_accuracy"] for row in results]
    x = np.arange(len(results))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, valid_f1, width, label="Best Valid Macro F1")
    ax.bar(x, test_f1, width, label="Test Macro F1")
    ax.bar(x + width, test_acc, width, label="Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x, names)
    ax.set_title("Experiment2: RGB vs predicted keypoint downstream comparison")
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "downstream_comparison.png", dpi=150)
    plt.close(fig)


def read_history_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {key: float(value) if key != "epoch" else int(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def save_downstream_learning_curve_overlay(output_dir: Path) -> None:
    histories = {
        "RGB Only": read_history_csv(output_dir / "metrics" / "rgb_only_history.csv"),
        "Predicted Keypoint Only": read_history_csv(output_dir / "metrics" / "pred_keypoint_only_history.csv"),
        "Predicted Keypoint + RGB Fusion": read_history_csv(
            output_dir / "metrics" / "pred_keypoint_fusion_history.csv"
        ),
    }
    colors = {
        "RGB Only": "#2ca02c",
        "Predicted Keypoint Only": "#1f77b4",
        "Predicted Keypoint + RGB Fusion": "#d62728",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    max_epoch = max(max(row["epoch"] for row in history) for history in histories.values())

    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        color = colors[name]
        axes[0].plot(epochs, [row["train_loss"] for row in history], "--", color=color, alpha=0.55)
        axes[0].plot(epochs, [row["valid_loss"] for row in history], "-", color=color, label=name)
        axes[1].plot(epochs, [row["train_macro_f1"] for row in history], "--", color=color, alpha=0.55)
        axes[1].plot(epochs, [row["valid_macro_f1"] for row in history], "-", color=color, label=name)
        axes[2].plot(epochs, [row["train_accuracy"] for row in history], "--", color=color, alpha=0.55)
        axes[2].plot(epochs, [row["valid_accuracy"] for row in history], "-", color=color, label=name)

    axes[0].set_title("Loss")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[1].set_title("Macro F1")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[2].set_title("Accuracy")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.0)

    for ax in axes:
        if max_epoch <= 1:
            ax.set_xlim(0.5, 1.5)
        else:
            ax.set_xlim(1, max_epoch)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Experiment2 downstream learning curves (solid=valid, dashed=train)")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "downstream_learning_curves_shared_axes.png", dpi=150)
    plt.close(fig)


def save_summary_markdown(
    args,
    pose_result: dict,
    classifier_results: list[dict],
    output_dir: Path,
    smoke: bool,
) -> None:
    result_by_mode = {row["mode"]: row for row in classifier_results}
    fusion_vs_rgb = (
        result_by_mode["pred_keypoint_fusion"]["test_macro_f1"]
        - result_by_mode["rgb_only"]["test_macro_f1"]
    )
    fusion_vs_keypoint = (
        result_by_mode["pred_keypoint_fusion"]["test_macro_f1"]
        - result_by_mode["pred_keypoint_only"]["test_macro_f1"]
    )
    lines = [
        "# Experiment2 후속 실험 결과",
        "",
        "## 실험 목적",
        "",
        "라벨링된 XML keypoint를 GT로 사용해 RGB 이미지만으로 keypoint를 예측하는 모델을 학습한 뒤,",
        "RGB-only, 예측 keypoint-only, RGB + 예측 keypoint 멀티모달 분류기를 비교했다.",
        "",
        "## 설정",
        "",
        f"- 실행 모드: {'smoke' if smoke else 'full'}",
        f"- manifest: `{args.manifest}`",
        f"- 입력 프레임 수: {args.num_frames if not smoke else min(args.num_frames, 4)}",
        f"- image size: {args.image_size}",
        f"- pose estimator epochs: {args.smoke_epochs if smoke else args.pose_epochs}",
        f"- classifier epochs: {args.smoke_epochs if smoke else args.classifier_epochs}",
        f"- batch size: {min(args.batch_size, 2) if smoke else args.batch_size}",
        "",
        "## Image -> Keypoint 예측기",
        "",
        f"- best epoch: {pose_result['best_epoch']}",
        f"- best valid normalized MPJPE: {pose_result['best_valid_mpjpe_norm']:.4f}",
        f"- checkpoint: `{pose_result['checkpoint_path']}`",
        "",
        "## Downstream 분류 비교",
        "",
        "| Model | Best Epoch | Best Valid Macro F1 | Test Accuracy | Test Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "rgb_only": "RGB Only",
        "pred_keypoint_only": "RGB -> Predicted Keypoint Only",
        "pred_keypoint_fusion": "RGB + RGB -> Predicted Keypoint Fusion",
    }
    for row in classifier_results:
        label = labels[row["mode"]]
        lines.append(
            f"| {label} | {row['best_epoch']} | {row['best_valid_macro_f1']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            f"- Fusion의 Test Macro F1 변화량 vs RGB-only: {fusion_vs_rgb:+.4f}",
            f"- Fusion의 Test Macro F1 변화량 vs predicted keypoint-only: {fusion_vs_keypoint:+.4f}",
            "- 이 실험은 GT keypoint를 직접 입력한 기존 baseline보다 실제 추론 환경에 가깝다.",
            "- 성능이 낮게 나오더라도 이는 keypoint 예측 오차가 downstream 분류에 누적되기 때문이다.",
            "- RGB-only보다 predicted keypoint-only가 높으면 자세 정보 추정이 행동 분류에 유효하다고 해석한다.",
            "- Fusion이 두 단일 입력 모델보다 높으면 RGB 정보와 예측 keypoint가 서로 보완한다고 해석한다.",
            "- Fusion이 낮거나 비슷하면 pose estimator 품질, fusion 구조, end-to-end fine-tuning 개선이 필요하다고 해석한다.",
            "",
            "## 산출물",
            "",
            "- `metrics/image_keypoint_estimator_history.csv`",
            "- `metrics/rgb_only_history.csv`",
            "- `metrics/pred_keypoint_only_history.csv`",
            "- `metrics/pred_keypoint_fusion_history.csv`",
            "- `metrics/experiment2_summary.json`",
            "- `figures/*learning_curve.png`",
            "- `figures/downstream_learning_curves_shared_axes.png`",
            "- `figures/*confusion.png`",
            "- `figures/downstream_comparison.png`",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/frames_224_trainvaltest.csv")
    parser.add_argument("--output-dir", default="outputs/experiment2")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pose-epochs", type=int, default=8)
    parser.add_argument("--classifier-epochs", type=int, default=15)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--pose-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--smoke-limit-per-class", type=int, default=2)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "checkpoints")
    save_json(vars(args), output_dir / "metrics" / "run_config.json")

    device = torch.device(device_name())
    print(f"device={device}")
    loaders = {
        "train": make_loader(args, "train", True, args.smoke),
        "val": make_loader(args, "val", False, args.smoke),
        "test": make_loader(args, "test", False, args.smoke),
    }
    print(
        "dataset sizes: "
        f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}"
    )

    pose_result = train_pose_estimator(args, loaders, device, output_dir, args.smoke)
    pose_estimator = ImageKeypointEstimator(
        pretrained=False,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    load_pose_checkpoint(pose_estimator, pose_result["checkpoint_path"], device)

    classifier_results = [
        train_classifier(args, None, loaders, device, output_dir, "rgb_only", args.smoke),
        train_classifier(args, pose_estimator, loaders, device, output_dir, "pred_keypoint_only", args.smoke),
        train_classifier(args, pose_estimator, loaders, device, output_dir, "pred_keypoint_fusion", args.smoke),
    ]
    save_comparison_figure(classifier_results, output_dir)
    save_downstream_learning_curve_overlay(output_dir)

    summary = {
        "pose_estimator": {k: v for k, v in pose_result.items() if k != "model"},
        "classifiers": classifier_results,
    }
    save_json(summary, output_dir / "metrics" / "experiment2_summary.json")
    with (output_dir / "metrics" / "downstream_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mode", "best_epoch", "best_valid_macro_f1", "test_accuracy", "test_macro_f1", "checkpoint_path"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "mode": row["mode"],
                "best_epoch": row["best_epoch"],
                "best_valid_macro_f1": row["best_valid_macro_f1"],
                "test_accuracy": row["test_accuracy"],
                "test_macro_f1": row["test_macro_f1"],
                "checkpoint_path": row["checkpoint_path"],
            }
            for row in classifier_results
        )
    save_summary_markdown(args, pose_result, classifier_results, output_dir, args.smoke)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote summary: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
