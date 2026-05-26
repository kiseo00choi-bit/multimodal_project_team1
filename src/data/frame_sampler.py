from __future__ import annotations

import numpy as np


def uniform_indices(start: int, end: int, num_frames: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if end < start:
        end = start
    values = np.linspace(start, end, num_frames)
    return [int(round(v)) for v in values]
