from __future__ import annotations

import argparse
import json

from multimodal_emotion.data.splits import create_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped train/validation folds.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-stratify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = create_group_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        labels=args.labels,
        n_splits=args.n_splits,
        stratified=not args.no_stratify,
    )
    print(json.dumps([{"train": str(train), "val": str(val)} for train, val in folds], indent=2))


if __name__ == "__main__":
    main()
