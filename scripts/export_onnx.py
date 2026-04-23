from __future__ import annotations

import argparse

from multimodal_emotion.training.runtime import ensure_training_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained multimodal emotion model to ONNX.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    ensure_training_dependencies()
    import torch
    from transformers import AutoTokenizer

    from multimodal_emotion.config import load_config
    from multimodal_emotion.export import export_model_to_onnx
    from multimodal_emotion.models import MultimodalEmotionModel

    args = parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    model = MultimodalEmotionModel(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    export_path = export_model_to_onnx(model, tokenizer, config, args.output_path)
    print(str(export_path))


if __name__ == "__main__":
    main()
