from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


RAW_ROOT = Path("data/raw/01-1.정식개방데이터")
OUT_ROOT = Path("data/extracted")


def class_name_from_zip(path: Path) -> str:
    return path.stem.split("_")[-1]


def target_kind(path: Path) -> str:
    parts = set(path.parts)
    if "01.원천데이터" in parts:
        return "videos"
    if "02.라벨링데이터" in parts:
        return "labels"
    raise ValueError(f"Cannot infer target kind from {path}")


def split_name(path: Path) -> str:
    parts = path.parts
    if "Training" in parts:
        return "Training"
    if "Validation" in parts:
        return "Validation"
    raise ValueError(f"Cannot infer split from {path}")


def safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/").lstrip("/")
    return Path(normalized).name


def extract_zip(zip_path: Path, out_root: Path, overwrite: bool = False) -> tuple[int, int]:
    split = split_name(zip_path)
    kind = target_kind(zip_path)
    class_name = class_name_from_zip(zip_path)
    target_dir = out_root / split / kind / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            filename = safe_member_name(info.filename)
            if not filename:
                continue
            target_path = target_dir / filename
            if target_path.exists() and target_path.stat().st_size == info.file_size and not overwrite:
                skipped += 1
                continue
            with archive.open(info) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted += 1
    return extracted, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zip_paths = sorted(args.raw_root.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No zip files found under {args.raw_root}")

    total_size = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            total_size += sum(info.file_size for info in archive.infolist())
    print(f"Found {len(zip_paths)} zip files.")
    print(f"Total uncompressed size: {total_size / 1024**3:.2f} GB")

    total_extracted = 0
    total_skipped = 0
    for index, zip_path in enumerate(zip_paths, start=1):
        print(f"[{index}/{len(zip_paths)}] {zip_path}")
        extracted, skipped = extract_zip(zip_path, args.out_root, overwrite=args.overwrite)
        total_extracted += extracted
        total_skipped += skipped
        print(f"  extracted={extracted} skipped={skipped}")

    print(f"Done. extracted={total_extracted} skipped={total_skipped}")


if __name__ == "__main__":
    main()
