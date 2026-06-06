from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AbnormalBehaviorDataset
from src.models.build import build_model
from src.train import classification_metrics, forward_batch, model_modalities, save_confusion
from src.utils import device_name, save_json


def evaluate_checkpoint(checkpoint_path: str | Path, split: str = "test") -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    use_frames, use_keypoints = model_modalities(config["model"])

    dataset = AbnormalBehaviorDataset(
        manifest_path=config.get("frame_manifest_path", "data/processed/frames_224_manifest.csv"),
        split=split,
        num_frames=int(config.get("num_frames", 16)),
        use_frames=use_frames,
        use_keypoints=use_keypoints,
        image_size=int(config.get("image_size", 224)),
        action_only=bool(config.get("action_only", True)),
        limit_per_class=config.get("eval_limit_per_class"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(device_name())
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    labels_all = []
    preds_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {config['experiment_name']} {split}", leave=False):
            logits, labels = forward_batch(model, batch, device)
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(labels_all, preds_all, num_classes=8)
    run_name = config["experiment_name"]
    suffix = split.lower()
    result = {
        "run_name": run_name,
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_valid_macro_f1": float(checkpoint.get("valid_macro_f1", -1.0)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "class_precision": metrics["precision"].tolist(),
        "class_recall": metrics["recall"].tolist(),
        "class_f1": metrics["f1"].tolist(),
        "class_support": metrics["support"].tolist(),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    output_root = Path(config.get("output_root", "outputs"))
    out_path = output_root / "metrics" / f"{run_name}_{suffix}_best_eval.json"
    save_json(result, out_path)
    save_confusion(
        labels_all,
        preds_all,
        output_root / "figures" / f"{run_name}_{suffix}_best_confusion.png",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, split=args.split)
    print(result)


if __name__ == "__main__":
    main()
