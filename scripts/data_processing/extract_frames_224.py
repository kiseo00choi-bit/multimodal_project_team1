from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


MANIFEST_PATH = Path("data/splits/manifest.csv")
OUT_ROOT = Path("data/processed/frames_224")
VIDEO_MANIFEST_PATH = Path("data/processed/frames_224_manifest.csv")
FRAME_INDEX_PATH = Path("data/processed/frame_index_224.csv")
LABEL_MAP_PATH = Path("data/processed/label_map.json")

CLASS_MAP = {
    "07": {"class_id": 0, "label_en": "fall", "label_ko": "전도"},
    "08": {"class_id": 1, "label_en": "broken", "label_ko": "파손"},
    "09": {"class_id": 2, "label_en": "fire", "label_ko": "방화"},
    "10": {"class_id": 3, "label_en": "smoke", "label_ko": "흡연"},
    "11": {"class_id": 4, "label_en": "abandon", "label_ko": "유기"},
    "12": {"class_id": 5, "label_en": "theft", "label_ko": "절도"},
    "13": {"class_id": 6, "label_en": "fight", "label_ko": "폭행"},
    "14": {"class_id": 7, "label_en": "weak_pedestrian", "label_ko": "교통약자"},
}


def class_code_from_label(label: str) -> str:
    return label.split(".", 1)[0]


def xml_meta(xml_path: Path) -> dict[str, int | None]:
    root = ET.parse(xml_path).getroot()
    task = root.find("./meta/task")
    xml_size = int(task.findtext("size")) if task is not None and task.findtext("size") else None
    xml_start = int(task.findtext("start_frame")) if task is not None and task.findtext("start_frame") else None
    xml_stop = int(task.findtext("stop_frame")) if task is not None and task.findtext("stop_frame") else None

    action_start = None
    action_end = None
    for track in root.findall("track"):
        label = (track.get("label") or "").lower()
        frames = []
        for child in track:
            if child.tag not in {"box", "points"}:
                continue
            if child.get("outside") == "1":
                continue
            frame = child.get("frame")
            if frame is not None:
                frames.append(int(frame))
        if not frames:
            continue
        if label.endswith("_start"):
            action_start = min(frames) if action_start is None else min(action_start, min(frames))
        elif label.endswith("_end"):
            action_end = max(frames) if action_end is None else max(action_end, max(frames))

    return {
        "xml_size": xml_size,
        "xml_start_frame": xml_start,
        "xml_stop_frame": xml_stop,
        "action_start_frame": action_start,
        "action_end_frame": action_end,
    }


def extract_video_frames(
    video_path: Path,
    frame_dir: Path,
    image_size: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if overwrite and frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(frame_dir.glob("frame_*.jpg")))
    if existing == expected_frames and expected_frames > 0 and not overwrite:
        cap.release()
        return existing, fps, width, height

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        out_path = frame_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        frame_index += 1

    cap.release()
    return frame_index, fps, width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[: args.limit]

    LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_PATH.write_text(
        json.dumps(CLASS_MAP, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    video_rows = []
    frame_rows = []

    for row_index, row in enumerate(rows, start=1):
        split = row["split"]
        raw_label = row["label"]
        class_code = class_code_from_label(raw_label)
        class_info = CLASS_MAP[class_code]
        video_path = Path(row["video_path"])
        xml_path = Path(row["xml_path"])
        frame_dir = (
            args.out_root
            / split
            / f"{class_code}_{class_info['label_en']}"
            / video_path.stem
        )

        meta = xml_meta(xml_path)
        frame_count, fps, width, height = extract_video_frames(
            video_path=video_path,
            frame_dir=frame_dir,
            image_size=args.image_size,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )

        action_start = meta["action_start_frame"]
        action_end = meta["action_end_frame"]
        if action_start is None:
            action_start = 0
        if action_end is None:
            action_end = frame_count - 1
        action_start = max(0, min(int(action_start), frame_count - 1))
        action_end = max(action_start, min(int(action_end), frame_count - 1))

        video_record = {
            "split": split,
            "class_id": class_info["class_id"],
            "class_code": class_code,
            "label_en": class_info["label_en"],
            "label_ko": class_info["label_ko"],
            "video_stem": video_path.stem,
            "video_path": video_path.as_posix(),
            "xml_path": xml_path.as_posix(),
            "frame_dir": frame_dir.as_posix(),
            "fps": f"{fps:.3f}",
            "source_width": width,
            "source_height": height,
            "image_size": args.image_size,
            "frame_count": frame_count,
            "xml_size": meta["xml_size"],
            "xml_start_frame": meta["xml_start_frame"],
            "xml_stop_frame": meta["xml_stop_frame"],
            "action_start_frame": action_start,
            "action_end_frame": action_end,
            "action_frame_count": action_end - action_start + 1,
        }
        video_rows.append(video_record)

        for frame_index in range(frame_count):
            frame_rows.append(
                {
                    "split": split,
                    "class_id": class_info["class_id"],
                    "class_code": class_code,
                    "label_en": class_info["label_en"],
                    "video_stem": video_path.stem,
                    "frame_index": frame_index,
                    "image_path": (frame_dir / f"frame_{frame_index:06d}.jpg").as_posix(),
                    "xml_path": xml_path.as_posix(),
                    "in_action_segment": int(action_start <= frame_index <= action_end),
                }
            )

        print(
            f"[{row_index}/{len(rows)}] {split} {class_code} {video_path.stem}: "
            f"frames={frame_count}, action={action_start}-{action_end}, fps={fps:.3f}"
        )

    VIDEO_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VIDEO_MANIFEST_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(video_rows)

    with FRAME_INDEX_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    print(f"wrote={VIDEO_MANIFEST_PATH}")
    print(f"wrote={FRAME_INDEX_PATH}")
    print(f"videos={len(video_rows)} frames={len(frame_rows)}")


if __name__ == "__main__":
    main()
