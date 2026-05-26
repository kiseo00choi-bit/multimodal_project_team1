from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.xml_parser import JOINT_NAMES, load_sampled_keypoints


MANIFEST_PATH = Path("data/processed/frames_224_trainvaltest.csv")
OUT_DIR = Path("docs/assets/examples")
KEYPOINT_CSV = OUT_DIR / "keypoint_examples.csv"

LABELS_KO = {
    "07": "전도",
    "08": "파손",
    "09": "방화",
    "10": "흡연",
    "11": "유기",
    "12": "절도",
    "13": "폭행",
    "14": "교통약자",
}

SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]


def action_midpoint(row: pd.Series) -> int:
    start = int(row.action_start_frame)
    end = int(row.action_end_frame)
    frame_count = int(row.frame_count)
    mid = (start + end) // 2
    return max(0, min(mid, frame_count - 1))


def draw_label(draw: ImageDraw.ImageDraw, text: str) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, 224, 28), fill=(0, 0, 0))
    draw.text((8, 5), text, fill=(255, 255, 255), font=font)


def draw_skeleton(image: Image.Image, keypoints_norm: list[float]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    points = []
    for idx in range(0, len(keypoints_norm), 2):
        x_norm = float(keypoints_norm[idx])
        y_norm = float(keypoints_norm[idx + 1])
        if x_norm <= 0.0 and y_norm <= 0.0:
            points.append(None)
        else:
            points.append((x_norm * image.width, y_norm * image.height))

    for start, end in SKELETON_EDGES:
        p1 = points[start]
        p2 = points[end]
        if p1 is not None and p2 is not None:
            draw.line((p1, p2), fill=(0, 255, 255), width=3)

    for point in points:
        if point is None:
            continue
        x, y = point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 64, 64), outline=(255, 255, 255))
    return overlay


def make_comparison_image(row: pd.Series) -> tuple[Path, list[float], int]:
    frame_idx = action_midpoint(row)
    frame_path = Path(row.frame_dir) / f"frame_{frame_idx:06d}.jpg"
    image = Image.open(frame_path).convert("RGB")
    keypoints = load_sampled_keypoints(
        xml_path=row.xml_path,
        frame_indices=[frame_idx],
        frame_count=int(row.frame_count),
        width=int(row.source_width),
        height=int(row.source_height),
    )[0].tolist()
    overlay = draw_skeleton(image, keypoints)

    canvas = Image.new("RGB", (448, 252), color=(245, 245, 245))
    canvas.paste(image, (0, 28))
    canvas.paste(overlay, (224, 28))
    draw = ImageDraw.Draw(canvas)
    class_code = str(row.class_code).zfill(2)
    title = f"{class_code} {row.label_en} / frame {frame_idx}"
    draw_label(draw, title)
    draw.text((70, 232), "RGB frame", fill=(40, 40, 40))
    draw.text((284, 232), "Keypoint overlay", fill=(40, 40, 40))

    out_path = OUT_DIR / f"{class_code}_{row.label_en}_example.png"
    canvas.save(out_path)
    return out_path, keypoints, frame_idx


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"class_code": str})
    rows = (
        manifest[manifest["split"] == "test"]
        .sort_values(["class_id", "video_stem"])
        .groupby("class_id", group_keys=False)
        .head(1)
        .sort_values("class_id")
    )

    csv_rows = []
    for _, row in rows.iterrows():
        image_path, keypoints, frame_idx = make_comparison_image(row)
        record = {
            "class_id": int(row.class_id),
            "class_code": str(row.class_code).zfill(2),
            "label_en": row.label_en,
            "label_ko": LABELS_KO.get(str(row.class_code).zfill(2), row.label_en),
            "video_stem": row.video_stem,
            "frame_index": frame_idx,
            "image_path": image_path.as_posix(),
        }
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            key = joint_name.lower().replace(" ", "_")
            record[f"{key}_x"] = round(float(keypoints[joint_idx * 2]), 6)
            record[f"{key}_y"] = round(float(keypoints[joint_idx * 2 + 1]), 6)
        csv_rows.append(record)
        print(f"wrote={image_path}")

    with KEYPOINT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote={KEYPOINT_CSV}")


if __name__ == "__main__":
    main()
