from __future__ import annotations

import argparse

from multimodal_emotion.training.runtime import ensure_training_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local multimodal video upload demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_training_dependencies("gradio", "librosa", "soundfile", "cv2", "imageio_ffmpeg", "PIL")

    from multimodal_emotion.demo.service import MultimodalDemoAnalyzer
    from multimodal_emotion.demo.ui import APP_CSS, build_demo

    args = parse_args()
    if args.preload_models:
        MultimodalDemoAnalyzer().preload_models()

    app = build_demo()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
