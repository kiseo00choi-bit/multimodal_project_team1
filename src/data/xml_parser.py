from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "Pelvis",
    "Left hip",
    "Left knee",
    "Left foot",
    "Right hip",
    "Right knee",
    "Right foot",
    "Spine naval",
    "Spine chest",
    "Neck base",
    "Center head",
    "Left shoulder",
    "Left elbow",
    "Left hand",
    "Right shoulder",
    "Right elbow",
    "Right hand",
]

JOINT_TO_INDEX = {name.lower().replace(" ", ""): idx for idx, name in enumerate(JOINT_NAMES)}


def _norm_label(label: str) -> str:
    return label.lower().replace(" ", "")


@lru_cache(maxsize=8192)
def parse_xml_keypoints(xml_path: str, frame_count: int) -> np.ndarray:
    keypoints = np.zeros((frame_count, len(JOINT_NAMES), 2), dtype=np.float32)
    root = ET.parse(xml_path).getroot()

    for track in root.findall("track"):
        label = _norm_label(track.get("label") or "")
        if label not in JOINT_TO_INDEX:
            continue
        joint_idx = JOINT_TO_INDEX[label]
        for points in track.findall("points"):
            if points.get("outside") == "1":
                continue
            frame_text = points.get("frame")
            point_text = points.get("points")
            if frame_text is None or point_text is None:
                continue
            frame_idx = int(frame_text)
            if frame_idx < 0 or frame_idx >= frame_count:
                continue
            xy = point_text.split(",")
            if len(xy) != 2:
                continue
            keypoints[frame_idx, joint_idx, 0] = float(xy[0])
            keypoints[frame_idx, joint_idx, 1] = float(xy[1])

    return keypoints.reshape(frame_count, -1)


def load_sampled_keypoints(
    xml_path: str | Path,
    frame_indices: list[int],
    frame_count: int,
    width: int,
    height: int,
) -> np.ndarray:
    full = parse_xml_keypoints(str(xml_path), int(frame_count))
    sampled = full[frame_indices].copy()
    sampled[:, 0::2] /= max(float(width), 1.0)
    sampled[:, 1::2] /= max(float(height), 1.0)
    return sampled.astype(np.float32)
