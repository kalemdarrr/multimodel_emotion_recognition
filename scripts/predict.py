from __future__ import annotations

import argparse
import json

from multimodal_emotion.training.runtime import ensure_training_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-off inference with the multimodal emotion model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--audio-features")
    parser.add_argument("--video-features")
    return parser.parse_args()


def main() -> None:
    ensure_training_dependencies()
    from multimodal_emotion.inference import load_model_for_inference, predict_single

    args = parse_args()
    config, tokenizer, model, device = load_model_for_inference(args.config, args.checkpoint)
    prediction = predict_single(
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        text=args.text,
        audio_features_path=args.audio_features,
        video_features_path=args.video_features,
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
