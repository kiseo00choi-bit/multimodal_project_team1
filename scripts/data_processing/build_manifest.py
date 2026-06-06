from __future__ import annotations

import csv
from pathlib import Path


EXTRACTED_ROOT = Path("data/extracted")
OUT_PATH = Path("data/splits/manifest.csv")


def iter_samples(root: Path):
    for split in ("Training", "Validation"):
        video_root = root / split / "videos"
        label_root = root / split / "labels"
        for video_path in sorted(video_root.rglob("*.mp4")):
            class_name = video_path.parent.name
            xml_path = label_root / class_name / f"{video_path.stem}.xml"
            yield {
                "split": split,
                "label": class_name,
                "video_path": video_path.as_posix(),
                "xml_path": xml_path.as_posix(),
                "has_xml": str(xml_path.exists()).lower(),
            }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_samples(EXTRACTED_ROOT))
    with OUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "label", "video_path", "xml_path", "has_xml"],
        )
        writer.writeheader()
        writer.writerows(rows)

    missing = sum(row["has_xml"] != "true" for row in rows)
    print(f"wrote={OUT_PATH}")
    print(f"samples={len(rows)} missing_xml={missing}")


if __name__ == "__main__":
    main()
