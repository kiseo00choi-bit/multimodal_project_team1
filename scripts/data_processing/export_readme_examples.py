from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.xml_parser import JOINT_NAMES, load_sampled_keypoints, parse_xml_keypoints


MANIFEST_PATH = Path("data/processed/frames_224_trainvaltest.csv")
OUT_DIR = Path("docs/assets/examples")
XML_SNIPPET_DIR = OUT_DIR / "xml_snippets"
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


def representative_frame(row: pd.Series) -> int:
    midpoint = action_midpoint(row)
    keypoints = parse_xml_keypoints(str(row.xml_path), int(row.frame_count)).reshape(int(row.frame_count), -1, 2)
    start = max(0, min(int(row.action_start_frame), int(row.frame_count) - 1))
    end = max(start, min(int(row.action_end_frame), int(row.frame_count) - 1))
    best_frame = midpoint
    best_score = -1
    best_distance = int(row.frame_count)
    for frame_idx in range(start, end + 1):
        visible_count = int(((keypoints[frame_idx, :, 0] > 0) | (keypoints[frame_idx, :, 1] > 0)).sum())
        distance = abs(frame_idx - midpoint)
        if visible_count > best_score or (visible_count == best_score and distance < best_distance):
            best_frame = frame_idx
            best_score = visible_count
            best_distance = distance
    if best_score == 0:
        xml_counts = xml_visible_element_counts(row.xml_path)
        for frame_idx, visible_count in xml_counts.items():
            distance = abs(frame_idx - midpoint)
            if visible_count > best_score or (visible_count == best_score and distance < best_distance):
                best_frame = frame_idx
                best_score = visible_count
                best_distance = distance
    return best_frame


def visible_keypoint_count(row: pd.Series, frame_idx: int) -> int:
    keypoints = parse_xml_keypoints(str(row.xml_path), int(row.frame_count)).reshape(int(row.frame_count), -1, 2)
    return int(((keypoints[frame_idx, :, 0] > 0) | (keypoints[frame_idx, :, 1] > 0)).sum())


def select_representatives(manifest: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    candidates = manifest[manifest["split"] == "test"].sort_values(["class_id", "video_stem"])
    for class_id, group in candidates.groupby("class_id", sort=True):
        best_row = None
        best_frame = -1
        best_score = -1
        for _, row in group.iterrows():
            frame_idx = representative_frame(row)
            score = visible_keypoint_count(row, frame_idx)
            if score > best_score:
                best_row = row.copy()
                best_frame = frame_idx
                best_score = score
            if best_score >= len(JOINT_NAMES):
                break
        if best_row is None:
            raise RuntimeError(f"No representative sample for class_id={class_id}")
        best_row["representative_frame"] = best_frame
        selected_rows.append(best_row)
    return pd.DataFrame(selected_rows).sort_values("class_id")


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
    frame_idx = int(row.get("representative_frame", representative_frame(row)))
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


def xml_points_for_frame(xml_path: str | Path, frame_idx: int) -> list[tuple[str, str, dict[str, str]]]:
    root = ET.parse(xml_path).getroot()
    records = []
    for track in root.findall("track"):
        label = track.get("label") or ""
        if label not in JOINT_NAMES:
            continue
        for point in track.findall("points"):
            if point.get("frame") != str(frame_idx):
                continue
            if point.get("outside") == "1":
                continue
            attributes = {key: value for key, value in point.attrib.items() if value is not None}
            records.append((label, track.get("id") or "", attributes))
            break
    order = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    return sorted(records, key=lambda item: order.get(item[0], 999))


def xml_visible_element_counts(xml_path: str | Path) -> dict[int, int]:
    root = ET.parse(xml_path).getroot()
    counts: dict[int, int] = {}
    for track in root.findall("track"):
        for element in list(track):
            if element.tag not in {"points", "box"}:
                continue
            if element.get("outside") == "1":
                continue
            frame_text = element.get("frame")
            if frame_text is None:
                continue
            frame_idx = int(frame_text)
            counts[frame_idx] = counts.get(frame_idx, 0) + 1
    return counts


def xml_elements_for_frame(xml_path: str | Path, frame_idx: int) -> list[tuple[str, str, ET.Element]]:
    root = ET.parse(xml_path).getroot()
    records = []
    for track in root.findall("track"):
        label = track.get("label") or ""
        for element in list(track):
            if element.tag not in {"points", "box"}:
                continue
            if element.get("frame") != str(frame_idx):
                continue
            if element.get("outside") == "1":
                continue
            records.append((label, track.get("id") or "", element))
    order = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    return sorted(records, key=lambda item: (order.get(item[0], 999), item[0], item[1]))


def xml_element_text(element: ET.Element) -> list[str]:
    attributes = {key: value for key, value in element.attrib.items() if value is not None}
    attr_text = " ".join(f'{key}="{value}"' for key, value in attributes.items())
    if len(element) == 0:
        return [f"    <{element.tag} {attr_text} />"]
    lines = [f"    <{element.tag} {attr_text}>"]
    for child in element:
        child_attr = " ".join(f'{key}="{value}"' for key, value in child.attrib.items())
        text = (child.text or "").strip()
        if child_attr:
            lines.append(f"      <{child.tag} {child_attr}>{text}</{child.tag}>")
        else:
            lines.append(f"      <{child.tag}>{text}</{child.tag}>")
    lines.append(f"    </{element.tag}>")
    return lines


def write_xml_snippet(row: pd.Series, frame_idx: int) -> Path:
    class_code = str(row.class_code).zfill(2)
    XML_SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = XML_SNIPPET_DIR / f"{class_code}_{row.label_en}_frame_{frame_idx}.xml"
    records = xml_elements_for_frame(row.xml_path, frame_idx)
    lines = [
        f"<!-- Extracted from {Path(row.xml_path).name} -->",
        f"<!-- video_stem={row.video_stem}, class={class_code} {row.label_en}, frame={frame_idx} -->",
        "<annotations_frame_sample>",
    ]
    for label, track_id, element in records:
        lines.append(f'  <track id="{track_id}" label="{label}">')
        lines.extend(xml_element_text(element))
        lines.append("  </track>")
    lines.append("</annotations_frame_sample>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"class_code": str})
    rows = select_representatives(manifest)

    csv_rows = []
    for _, row in rows.iterrows():
        image_path, keypoints, frame_idx = make_comparison_image(row)
        xml_snippet_path = write_xml_snippet(row, frame_idx)
        record = {
            "class_id": int(row.class_id),
            "class_code": str(row.class_code).zfill(2),
            "label_en": row.label_en,
            "label_ko": LABELS_KO.get(str(row.class_code).zfill(2), row.label_en),
            "video_stem": row.video_stem,
            "frame_index": frame_idx,
            "image_path": image_path.as_posix(),
            "xml_path": row.xml_path,
            "xml_snippet_path": xml_snippet_path.as_posix(),
        }
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            key = joint_name.lower().replace(" ", "_")
            record[f"{key}_x"] = round(float(keypoints[joint_idx * 2]), 6)
            record[f"{key}_y"] = round(float(keypoints[joint_idx * 2 + 1]), 6)
        csv_rows.append(record)
        print(f"wrote={image_path}")
        print(f"wrote={xml_snippet_path}")

    with KEYPOINT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote={KEYPOINT_CSV}")


if __name__ == "__main__":
    main()
