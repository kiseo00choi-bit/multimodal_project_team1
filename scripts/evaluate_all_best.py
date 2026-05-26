from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_checkpoint
from src.utils import save_json


FULL_CHECKPOINTS = [
    "outputs/checkpoints/baseline_cnn_avg_best.pt",
    "outputs/checkpoints/baseline_cnn_lstm_best.pt",
    "outputs/checkpoints/baseline_keypoint_best.pt",
    "outputs/checkpoints/fusion_best.pt",
]

SMOKE_CHECKPOINTS = [
    "outputs/checkpoints/baseline_cnn_avg_smoke_best.pt",
    "outputs/checkpoints/baseline_cnn_lstm_smoke_best.pt",
    "outputs/checkpoints/baseline_keypoint_smoke_best.pt",
    "outputs/checkpoints/fusion_smoke_best.pt",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    checkpoints = SMOKE_CHECKPOINTS if args.smoke else FULL_CHECKPOINTS
    results = []
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        if not path.exists():
            print(f"missing: {path}")
            continue
        results.append(evaluate_checkpoint(path, split="test"))
    suffix = "smoke_" if args.smoke else ""
    save_json({"results": results}, f"outputs/metrics/all_best_{suffix}test_eval.json")


if __name__ == "__main__":
    main()
