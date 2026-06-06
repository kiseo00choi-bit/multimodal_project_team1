from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_one
from src.utils import load_config, save_json
from src.evaluate import evaluate_checkpoint


CONFIGS = [
    "configs/baseline_cnn_avg.yaml",
    "configs/baseline_cnn_lstm.yaml",
    "configs/baseline_keypoint.yaml",
    "configs/fusion.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/experiment1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-frames", type=int)
    args = parser.parse_args()

    results = []
    test_results = []
    for config_path in CONFIGS:
        config = load_config(config_path)
        config["output_root"] = args.output_dir
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.num_frames is not None:
            config["num_frames"] = args.num_frames
        print(f"==== Running {config['experiment_name']} ====")
        train_result = train_one(config, smoke=args.smoke)
        results.append(train_result)

        checkpoint_path = train_result["checkpoint_path"]
        test_split = config.get("test_split", "test")
        print(f"==== Testing {config['experiment_name']} on {test_split} ====")
        test_results.append(evaluate_checkpoint(checkpoint_path, split=test_split))

    suffix = "smoke" if args.smoke else "full"
    save_json(
        {"train_results": results, "test_results": test_results},
        Path(args.output_dir) / "metrics" / f"all_experiments_{suffix}.json",
    )


if __name__ == "__main__":
    main()
