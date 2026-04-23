from __future__ import annotations

import argparse
import json

from multimodal_emotion.data.synthetic import build_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic multimodal emotion dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    )
    parser.add_argument("--audio-dim", type=int, default=64)
    parser.add_argument("--video-dim", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=56)
    parser.add_argument("--val-size", type=int, default=14)
    parser.add_argument("--test-size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_paths = build_synthetic_dataset(
        output_dir=args.output_dir,
        labels=args.labels,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(json.dumps({split: str(path) for split, path in manifest_paths.items()}, indent=2))


if __name__ == "__main__":
    main()
