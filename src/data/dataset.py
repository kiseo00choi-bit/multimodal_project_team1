from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.frame_sampler import uniform_indices
from src.data.preprocessing import image_transform
from src.data.xml_parser import load_sampled_keypoints


class AbnormalBehaviorDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path = "data/processed/frames_224_manifest.csv",
        split: str = "Training",
        num_frames: int = 16,
        use_frames: bool = True,
        use_keypoints: bool = False,
        image_size: int = 224,
        action_only: bool = True,
        limit_per_class: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.num_frames = num_frames
        self.use_frames = use_frames
        self.use_keypoints = use_keypoints
        self.action_only = action_only
        self.transform = image_transform(image_size)

        df = pd.read_csv(self.manifest_path, dtype={"class_code": str})
        df = df[df["split"] == split].copy()
        df["class_id"] = df["class_id"].astype(int)
        if limit_per_class is not None:
            df = (
                df.groupby("class_id", group_keys=False)
                .head(limit_per_class)
                .reset_index(drop=True)
            )
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        frame_count = int(row.frame_count)
        if self.action_only:
            start = int(row.action_start_frame)
            end = int(row.action_end_frame)
        else:
            start = 0
            end = frame_count - 1
        start = max(0, min(start, frame_count - 1))
        end = max(start, min(end, frame_count - 1))
        indices = uniform_indices(start, end, self.num_frames)

        sample = {
            "label": torch.tensor(int(row.class_id), dtype=torch.long),
            "metadata": {
                "video_stem": row.video_stem,
                "frame_indices": indices,
                "xml_path": row.xml_path,
                "frame_dir": row.frame_dir,
            },
        }

        if self.use_frames:
            frames = []
            frame_dir = Path(row.frame_dir)
            for frame_idx in indices:
                image_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
                with Image.open(image_path) as image:
                    frames.append(self.transform(image.convert("RGB")))
            sample["frames"] = torch.stack(frames, dim=0)

        if self.use_keypoints:
            keypoints = load_sampled_keypoints(
                xml_path=row.xml_path,
                frame_indices=indices,
                frame_count=frame_count,
                width=int(row.source_width),
                height=int(row.source_height),
            )
            sample["keypoints"] = torch.from_numpy(keypoints)

        return sample
