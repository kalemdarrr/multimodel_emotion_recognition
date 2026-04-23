from __future__ import annotations

import pandas as pd
import gradio as gr

from .service import MultimodalDemoAnalyzer


APP_CSS = """
:root {
  --ink: #18151c;
  --paper: #f7f1e8;
  --sand: #ead8bf;
  --gold: #b9822c;
  --rust: #a1492b;
  --teal: #1f5f66;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(185,130,44,0.18), transparent 34%),
    radial-gradient(circle at bottom right, rgba(31,95,102,0.16), transparent 28%),
    linear-gradient(135deg, #f7f1e8 0%, #fff9ef 45%, #efe2cc 100%);
  color: var(--ink);
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
}

.app-shell {
  max-width: 1180px;
  margin: 0 auto;
}

.hero-card, .result-card {
  border: 1px solid rgba(24, 21, 28, 0.10);
  background: rgba(255, 251, 244, 0.82);
  backdrop-filter: blur(8px);
  border-radius: 24px;
  box-shadow: 0 18px 40px rgba(76, 55, 31, 0.10);
}

.hero-card {
  padding: 28px 28px 16px 28px;
  margin-bottom: 18px;
}

.result-card {
  padding: 18px 22px;
}

.hero-card h1 {
  font-size: 2.3rem;
  margin-bottom: 0.25rem;
  color: #2f2214;
}

.hero-card p {
  font-size: 1.05rem;
  line-height: 1.6;
}

.gold-button {
  background: linear-gradient(120deg, var(--gold), #cf9b48 50%, #f0c472);
  color: #1d160d;
  border: none;
}
"""


def _format_summary(result: dict) -> str:
    label = result["predicted_label"].title()
    confidence = result["confidence"] * 100.0
    transcript_source = result["transcript_source"].title()
    return (
        f"## Final Emotion: **{label}**\n\n"
        f"Confidence: **{confidence:.1f}%**\n\n"
        f"Transcript source: **{transcript_source}**"
    )


def _format_notes(result: dict) -> str:
    audio_note = result["notes"]["audio"]
    transcript_note = result["notes"]["transcript"]
    return (
        "### Fusion Notes\n\n"
        f"- {audio_note}\n"
        f"- {transcript_note}\n"
        "- Final prediction uses report-aligned late fusion across the available modalities.\n"
        "- The fusion layer downweights weak or missing modalities instead of failing hard."
    )


def build_demo() -> gr.Blocks:
    analyzer = MultimodalDemoAnalyzer()

    def run_analysis(video_path: str | None, transcript_override: str) -> tuple[str, dict, pd.DataFrame, str, str, dict]:
        if not video_path:
            raise gr.Error("Upload a video file before running the analysis.")

        result = analyzer.analyze(video_path, transcript_override or None)
        probabilities = {label.title(): round(score, 4) for label, score in result["probabilities"].items()}
        modality_df = pd.DataFrame(result["modality_rows"])
        return (
            _format_summary(result),
            probabilities,
            modality_df,
            result["transcript"],
            _format_notes(result),
            result,
        )

    with gr.Blocks(title="Multimodal Emotion Demo") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="hero-card">
                  <h1>Multimodal Emotion Research Demo</h1>
                  <p>
                    Upload an English video clip and the system will analyze it with the report's
                    first-stage strategy: transcript ASR, text emotion scoring, speech emotion scoring,
                    visual emotion scoring, and confidence-weighted late fusion.
                  </p>
                </section>
                """
            )

            with gr.Row():
                with gr.Column(scale=5, elem_classes=["result-card"]):
                    video_input = gr.Video(label="Upload Video", sources=["upload"])
                    transcript_override = gr.Textbox(
                        label="Transcript Override (Optional)",
                        lines=5,
                        placeholder="Leave blank to let the app generate an English transcript automatically.",
                    )
                    analyze_button = gr.Button("Analyze Emotion", elem_classes=["gold-button"], variant="primary")
                with gr.Column(scale=4, elem_classes=["result-card"]):
                    summary_output = gr.Markdown("## Final Emotion\n\nUpload a video to begin.")
                    probabilities_output = gr.Label(label="Fused Emotion Scores", num_top_classes=7)

            with gr.Row():
                with gr.Column(scale=5, elem_classes=["result-card"]):
                    transcript_output = gr.Textbox(label="Transcript Used", lines=7)
                with gr.Column(scale=5, elem_classes=["result-card"]):
                    notes_output = gr.Markdown("### Fusion Notes")

            with gr.Row():
                modality_output = gr.Dataframe(
                    headers=["modality", "status", "top_emotion", "confidence", "quality", "fusion_weight", "note"],
                    datatype=["str", "str", "str", "number", "number", "number", "str"],
                    interactive=False,
                    label="Modality Breakdown",
                )

            with gr.Accordion("Debug Payload", open=False):
                raw_output = gr.JSON(label="Raw Analysis")

            analyze_button.click(
                fn=run_analysis,
                inputs=[video_input, transcript_override],
                outputs=[
                    summary_output,
                    probabilities_output,
                    modality_output,
                    transcript_output,
                    notes_output,
                    raw_output,
                ],
            )

    return demo
