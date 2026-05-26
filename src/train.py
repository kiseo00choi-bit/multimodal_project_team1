from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AbnormalBehaviorDataset
from src.models.build import build_model
from src.utils import device_name, ensure_dir, load_config, save_json, set_seed


def model_modalities(model_name: str) -> tuple[bool, bool]:
    if model_name in {"cnn_avg", "cnn_lstm"}:
        return True, False
    if model_name == "keypoint_lstm":
        return False, True
    if model_name in {"fusion", "fusion_attention"}:
        return True, True
    raise ValueError(model_name)


def make_loader(config: dict, split: str, smoke: bool) -> DataLoader:
    use_frames, use_keypoints = model_modalities(config["model"])
    limit = int(config.get("smoke_limit_per_class", 2)) if smoke else config.get("limit_per_class")
    dataset = AbnormalBehaviorDataset(
        manifest_path=config.get("frame_manifest_path", "data/processed/frames_224_manifest.csv"),
        split=split,
        num_frames=int(config.get("num_frames", 16)),
        use_frames=use_frames,
        use_keypoints=use_keypoints,
        image_size=int(config.get("image_size", 224)),
        action_only=bool(config.get("action_only", True)),
        limit_per_class=limit,
    )
    split_key = str(split).lower()
    should_shuffle = split_key in {"train", "training"}
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8 if not smoke else 2)),
        shuffle=should_shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def forward_batch(model: nn.Module, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frames = batch.get("frames")
    keypoints = batch.get("keypoints")
    if frames is not None:
        frames = frames.to(device, non_blocking=True)
    if keypoints is not None:
        keypoints = keypoints.to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    logits = model(frames=frames, keypoints=keypoints)
    return logits, labels


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train(train)
    losses = []
    preds = []
    labels_all = []
    desc = "train" if train else "valid"
    for batch in tqdm(loader, desc=desc, leave=False):
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits, labels = forward_batch(model, batch, device)
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
    }


def confusion_matrix_np(labels: list[int], preds: list[int], num_classes: int = 8) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        if 0 <= int(label) < num_classes and 0 <= int(pred) < num_classes:
            cm[int(label), int(pred)] += 1
    return cm


def classification_metrics(labels: list[int], preds: list[int], num_classes: int = 8) -> dict:
    cm = confusion_matrix_np(labels, preds, num_classes=num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    support = cm.sum(axis=1)

    for idx in range(num_classes):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        precision[idx] = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall[idx] = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1[idx] = (
            2 * precision[idx] * recall[idx] / (precision[idx] + recall[idx])
            if precision[idx] + recall[idx] > 0
            else 0.0
        )

    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
    }


def save_confusion(labels: list[int], preds: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix_np(labels, preds, num_classes=8)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            ax.text(x, y, str(cm[y, x]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_learning_curve(history: list[dict], path: Path, run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="train")
    axes[1].plot(epochs, [row["valid_macro_f1"] for row in history], label="val")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[2].plot(epochs, [row["valid_accuracy"] for row in history], label="val")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one(config: dict, smoke: bool = False) -> dict:
    set_seed(int(config.get("seed", 42)))
    if smoke:
        config = dict(config)
        config["epochs"] = int(config.get("smoke_epochs", 1))
        config["num_frames"] = int(config.get("smoke_num_frames", min(4, int(config.get("num_frames", 16)))))
        config["batch_size"] = int(config.get("smoke_batch_size", 2))
        config["pretrained"] = False

    run_name = config["experiment_name"] + ("_smoke" if smoke else "")
    run_dir = ensure_dir(Path("outputs") / "runs" / run_name)
    save_json(config, run_dir / "config.json")

    device = torch.device(device_name())
    model = build_model(config).to(device)
    train_loader = make_loader(config, config.get("train_split", "Training"), smoke)
    valid_loader = make_loader(config, config.get("val_split", "Validation"), smoke)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    best_f1 = -1.0
    history = []
    start_time = time.time()
    for epoch in range(1, int(config.get("epochs", 20)) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)
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
            f"{run_name} epoch {epoch}: "
            f"train_loss={row['train_loss']:.4f} train_f1={row['train_macro_f1']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_f1={row['valid_macro_f1']:.4f}"
        )
        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            ensure_dir("outputs/checkpoints")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "valid_macro_f1": best_f1,
                },
                Path("outputs/checkpoints") / f"{run_name}_best.pt",
            )
            save_confusion(
                valid_metrics["labels"],
                valid_metrics["preds"],
                Path("outputs/figures") / f"{run_name}_confusion.png",
            )

    ensure_dir("outputs/metrics")
    metrics_path = Path("outputs/metrics") / f"{run_name}_history.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_learning_curve(
        history,
        Path("outputs/figures") / f"{run_name}_learning_curve.png",
        run_name,
    )

    final_metrics = classification_metrics(valid_metrics["labels"], valid_metrics["preds"], num_classes=8)
    result = {
        "run_name": run_name,
        "best_valid_macro_f1": float(best_f1),
        "last_valid_accuracy": float(history[-1]["valid_accuracy"]),
        "last_valid_macro_f1": float(history[-1]["valid_macro_f1"]),
        "seconds": round(time.time() - start_time, 2),
        "class_precision": final_metrics["precision"].tolist(),
        "class_recall": final_metrics["recall"].tolist(),
        "class_f1": final_metrics["f1"].tolist(),
        "class_support": final_metrics["support"].tolist(),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(Path("outputs/checkpoints") / f"{run_name}_best.pt"),
    }
    save_json(result, Path("outputs/metrics") / f"{run_name}_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-frames", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_frames is not None:
        config["num_frames"] = args.num_frames
    result = train_one(config, smoke=args.smoke)
    print(result)


if __name__ == "__main__":
    main()
