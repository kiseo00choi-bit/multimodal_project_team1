from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment2.run_experiment2 import (  # noqa: E402
    CLASS_NAMES,
    ImageKeypointEstimator,
    RGBPredictedKeypointFusionClassifier,
)
from src.data.frame_sampler import uniform_indices  # noqa: E402
from src.data.preprocessing import image_transform  # noqa: E402
from src.utils import device_name  # noqa: E402


LABEL_KO = {
    0: "fall",
    1: "broken",
    2: "fire",
    3: "smoke",
    4: "abandon",
    5: "theft",
    6: "fight",
    7: "weak_pedestrian",
}


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()


def choose_sample(manifest: Path, split: str, class_id: int | None, sample_index: int) -> pd.Series:
    df = pd.read_csv(manifest, dtype={"class_code": str})
    df = df[df["split"] == split].copy()
    if class_id is not None:
        df = df[df["class_id"].astype(int) == class_id].copy()
    if df.empty:
        raise ValueError(f"No samples found for split={split!r}, class_id={class_id!r}")
    df = df.reset_index(drop=True)
    return df.iloc[min(sample_index, len(df) - 1)]


def predict_clip(
    row: pd.Series,
    pose_model: ImageKeypointEstimator,
    fusion_model: RGBPredictedKeypointFusionClassifier,
    device: torch.device,
    num_frames: int,
    image_size: int,
) -> tuple[np.ndarray, list[int]]:
    frame_count = int(row.frame_count)
    start = max(0, min(int(row.action_start_frame), frame_count - 1))
    end = max(start, min(int(row.action_end_frame), frame_count - 1))
    indices = uniform_indices(start, end, num_frames)
    transform = image_transform(image_size)

    frames = []
    frame_dir = Path(row.frame_dir)
    for frame_idx in indices:
        image_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
        with Image.open(image_path) as image:
            frames.append(transform(image.convert("RGB")))
    batch = torch.stack(frames, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_keypoints = pose_model(batch)
        logits = fusion_model(batch, pred_keypoints)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return probs, indices


def draw_panel(
    frame: np.ndarray,
    probs: np.ndarray,
    true_label: str,
    frame_idx: int,
    total_frames: int,
    action_start: int,
    action_end: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w = max(420, int(w * 0.34))
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (5, 18, 26), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    pred_id = int(np.argmax(probs))
    pred_label = LABEL_KO[pred_id]
    confidence = float(probs[pred_id])
    top3 = np.argsort(probs)[::-1][:3]

    x = w - panel_w + 28
    y = 44
    in_action = action_start <= frame_idx <= action_end
    status_color = (0, 220, 255) if in_action else (160, 160, 160)

    cv2.putText(frame, "LIVE CCTV MONITOR", (26, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 246), 2, cv2.LINE_AA)
    cv2.circle(frame, (30, 82), 9, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (48, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "ACTION SEGMENT" if in_action else "CONTEXT",
        (145, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(frame, "Abnormal Behavior", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (50, 255, 246), 2, cv2.LINE_AA)
    y += 46
    cv2.putText(frame, f"Prediction: {pred_label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    y += 36
    cv2.putText(frame, f"Confidence: {confidence:.3f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 240, 255), 2, cv2.LINE_AA)
    y += 36
    cv2.putText(frame, f"GT Label: {true_label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 2, cv2.LINE_AA)
    y += 40
    cv2.putText(frame, "Top-3 probabilities", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 200, 210), 2, cv2.LINE_AA)
    y += 28

    bar_x = x
    bar_w = panel_w - 64
    for class_id in top3:
        label = LABEL_KO[int(class_id)]
        value = float(probs[int(class_id)])
        cv2.putText(frame, f"{label:16s} {value:.3f}", (bar_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        y += 10
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 13), (25, 58, 68), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_w * value), y + 13), (25, 217, 210), -1)
        y += 28

    progress_w = panel_w - 64
    progress_y = h - 54
    cv2.putText(
        frame,
        f"Frame {frame_idx:03d}/{total_frames - 1:03d}",
        (x, progress_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 220, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (x, progress_y), (x + progress_w, progress_y + 10), (25, 58, 68), -1)
    cv2.rectangle(frame, (x, progress_y), (x + int(progress_w * frame_idx / max(1, total_frames - 1)), progress_y + 10), (50, 255, 246), -1)

    if in_action:
        cv2.rectangle(frame, (8, 8), (w - panel_w - 8, h - 8), (0, 220, 255), 3)
    return frame


def make_demo(args: argparse.Namespace) -> None:
    device = torch.device(device_name())
    row = choose_sample(Path(args.manifest), args.split, args.class_id, args.sample_index)

    pose_model = ImageKeypointEstimator(pretrained=False, freeze_backbone=True).to(device)
    fusion_model = RGBPredictedKeypointFusionClassifier(pretrained=False, freeze_backbone=True).to(device)
    load_checkpoint(pose_model, args.pose_checkpoint, device)
    load_checkpoint(fusion_model, args.fusion_checkpoint, device)

    probs, sampled_indices = predict_clip(row, pose_model, fusion_model, device, args.num_frames, args.image_size)

    video_path = Path(row.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or row.fps or 3.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or row.frame_count)
    out_fps = args.output_fps or min(max(source_fps * 2, 6.0), 12.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, args.max_width / max(1, width))
    out_size = (int(width * scale), int(height * scale))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, out_size)

    action_start = int(row.action_start_frame)
    action_end = int(row.action_end_frame)
    true_label = str(row.label_en)
    frame_idx = 0
    written = 0
    max_frames = min(total_frames, int(args.max_seconds * source_fps)) if args.max_seconds else total_frames
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        frame = draw_panel(frame, probs, true_label, frame_idx, total_frames, action_start, action_end)
        writer.write(frame)
        written += 1
        frame_idx += 1

    writer.release()
    cap.release()

    print(f"wrote demo: {output}")
    print(f"video: {video_path}")
    print(f"gt={true_label} pred={LABEL_KO[int(np.argmax(probs))]} conf={float(np.max(probs)):.4f}")
    print(f"sampled action frames: {sampled_indices}")
    print(f"frames written={written} fps={out_fps:.2f} size={out_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/frames_224_trainvaltest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--pose-checkpoint", default="outputs/experiment2/checkpoints/image_keypoint_estimator_best.pt")
    parser.add_argument("--fusion-checkpoint", default="outputs/experiment2/checkpoints/pred_keypoint_fusion_best.pt")
    parser.add_argument("--output", default="outputs/demo/cctv_realtime_demo.mp4")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    make_demo(parse_args())
