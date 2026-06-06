from __future__ import annotations

import argparse
import csv
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2


MANIFEST_PATH = Path("data/splits/manifest.csv")


def xml_frame_meta(xml_path: Path) -> tuple[int | None, int | None, int | None]:
    root = ET.parse(xml_path).getroot()
    task = root.find("./meta/task")
    if task is None:
        return None, None, None
    size = task.findtext("size")
    start = task.findtext("start_frame")
    stop = task.findtext("stop_frame")
    return (
        int(size) if size is not None else None,
        int(start) if start is not None else None,
        int(stop) if stop is not None else None,
    )


def video_meta(video_path: Path) -> tuple[float, int, float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[: args.limit]

    fps_values: list[float] = []
    frame_values: list[int] = []
    xml_sizes: list[int] = []
    mismatches = []
    by_split_label = defaultdict(list)

    for index, row in enumerate(rows, start=1):
        video_path = Path(row["video_path"])
        xml_path = Path(row["xml_path"])
        fps, frames, width, height = video_meta(video_path)
        xml_size, start, stop = xml_frame_meta(xml_path)

        fps_values.append(fps)
        frame_values.append(frames)
        if xml_size is not None:
            xml_sizes.append(xml_size)
        by_split_label[(row["split"], row["label"])].append((fps, frames, xml_size))

        expected_xml_size = stop - start + 1 if start is not None and stop is not None else None
        if xml_size != frames or expected_xml_size != frames:
            mismatches.append(
                {
                    "video": str(video_path),
                    "fps": fps,
                    "video_frames": frames,
                    "xml_size": xml_size,
                    "xml_start": start,
                    "xml_stop": stop,
                    "width": width,
                    "height": height,
                }
            )

        if index <= 8:
            seconds = frames / fps if fps else 0
            print(
                f"sample {index}: fps={fps:.3f}, frames={frames}, seconds={seconds:.2f}, "
                f"xml_size={xml_size}, xml_range={start}-{stop}, file={video_path.name}"
            )

    rounded_fps = Counter(round(value, 3) for value in fps_values)
    print()
    print(f"videos={len(rows)}")
    print(f"fps_distribution={dict(sorted(rounded_fps.items()))}")
    print(
        "frame_count: "
        f"min={min(frame_values)}, median={statistics.median(frame_values)}, "
        f"max={max(frame_values)}"
    )
    print(
        "duration_seconds: "
        f"min={min(f / fps for f, fps in zip(frame_values, fps_values)):.2f}, "
        f"median={statistics.median(f / fps for f, fps in zip(frame_values, fps_values)):.2f}, "
        f"max={max(f / fps for f, fps in zip(frame_values, fps_values)):.2f}"
    )
    print(f"xml_size_distribution={dict(sorted(Counter(xml_sizes).items()))}")
    print(f"frame_xml_mismatches={len(mismatches)}")

    print()
    print("by split/label:")
    for key in sorted(by_split_label):
        values = by_split_label[key]
        fps_set = sorted({round(item[0], 3) for item in values})
        frames_set = sorted({item[1] for item in values})
        print(
            f"{key[0]:10s} {key[1]:8s} count={len(values):4d} "
            f"fps={fps_set} frames={frames_set[:8]}{'...' if len(frames_set) > 8 else ''}"
        )

    if mismatches:
        print()
        print("first mismatches:")
        for item in mismatches[:10]:
            print(item)


if __name__ == "__main__":
    main()
