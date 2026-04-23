# Multimodal Emotion Recognition Project

This repository implements an English-only multimodal emotion recognition pipeline based on the PDF blueprint. The text branch uses standard BERT (`bert-base-uncased`), not BERTurk. The project is organized around a practical late-fusion system:

- Text branch: BERT on raw English utterances
- Audio branch: cached speech embeddings extracted from English audio
- Video branch: cached visual embeddings extracted from face or clip frames
- Fusion branch: gated late fusion with modality dropout for robustness

The repository is designed so you can move from raw data to training, evaluation, prediction, and ONNX export with a consistent manifest format.

## What Is Included

- A configurable training pipeline for multimodal classification
- Manifest validation and leak-aware split utilities
- Synthetic dataset generation for smoke testing
- Audio feature extraction with Wav2Vec2-style encoders
- Video feature extraction with EfficientNet-based frame pooling
- Evaluation utilities for macro-F1, weighted-F1, accuracy, calibration, and confusion matrices
- Prediction and ONNX export scripts
- A local upload-based demo UI for video emotion analysis

## Project Assumptions

- Language: English
- Primary task: multi-class emotion classification
- Default text encoder: `bert-base-uncased`
- Default labels:
  - `neutral`
  - `joy`
  - `sadness`
  - `anger`
  - `fear`
  - `disgust`
  - `surprise`
- Audio and video are trained through cached feature files for efficiency and reproducibility

## Repository Layout

```text
configs/
  default.json
  tiny.json
data/
  example_manifest.jsonl
scripts/
  build_synthetic_dataset.py
  create_group_splits.py
  evaluate.py
  export_onnx.py
  extract_audio_features.py
  extract_video_features.py
  predict.py
  run_demo.py
  train.py
  validate_manifest.py
src/multimodal_emotion/
  config.py
  data/
  demo/
  evaluation/
  export/
  inference/
  models/
  training/
tests/
  test_manifest.py
  test_metrics.py
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Manifest Format

Every sample is a JSON object on its own line. The required and optional fields are:

```json
{
  "sample_id": "train_0001",
  "label": "joy",
  "text": "I am really happy to see you today.",
  "group_id": "dialogue_12",
  "speaker_id": "speaker_04",
  "audio_path": "raw/audio/train_0001.wav",
  "video_path": "raw/video/train_0001.mp4",
  "audio_features_path": "features/audio/train_0001.npy",
  "video_features_path": "features/video/train_0001.npy",
  "metadata": {
    "split": "train",
    "transcript_source": "gold"
  }
}
```

Training requires `text`, `label`, `audio_features_path`, and `video_features_path`. Raw `audio_path` and `video_path` are used by the extraction scripts.

## Quick Start

Generate a synthetic dataset:

```bash
PYTHONPATH=src python3 scripts/build_synthetic_dataset.py --output-dir data/synthetic_demo
```

Validate the manifests:

```bash
PYTHONPATH=src python3 scripts/validate_manifest.py \
  --manifest data/synthetic_demo/train.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

Create grouped folds:

```bash
PYTHONPATH=src python3 scripts/create_group_splits.py \
  --manifest data/synthetic_demo/train.jsonl \
  --output-dir data/synthetic_demo/folds \
  --labels neutral joy sadness anger fear disgust surprise
```

Train the real model:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --config configs/default.json \
  --train-manifest data/processed/train.jsonl \
  --val-manifest data/processed/val.jsonl \
  --output-dir artifacts/run_001
```

Evaluate a checkpoint:

```bash
PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/default.json \
  --checkpoint artifacts/run_001/best_model.pt \
  --manifest data/processed/test.jsonl \
  --output-dir artifacts/eval_001
```

Predict one example:

```bash
PYTHONPATH=src python3 scripts/predict.py \
  --config configs/default.json \
  --checkpoint artifacts/run_001/best_model.pt \
  --text "I am feeling much better now." \
  --audio-features features/audio/example.npy \
  --video-features features/video/example.npy
```

Export to ONNX:

```bash
PYTHONPATH=src python3 scripts/export_onnx.py \
  --config configs/default.json \
  --checkpoint artifacts/run_001/best_model.pt \
  --output-path artifacts/run_001/model.onnx
```

## Demo UI

Launch the local upload interface:

```bash
PYTHONPATH=src python3 scripts/run_demo.py --host 127.0.0.1 --port 7860
```

The demo follows the report's stage-one recommendation:

- English ASR with Whisper
- Text emotion scoring with BERT
- Speech emotion scoring with Wav2Vec2
- Visual emotion scoring from sampled face frames
- Confidence-weighted late fusion across available modalities

You upload a video, the app extracts audio, generates or accepts an English transcript, scores each modality, and returns:

- the final fused emotion
- modality-by-modality predictions
- the transcript used for the analysis
- confidence-weighted fusion details

## Feature Extraction

Extract cached audio embeddings from raw waveforms:

```bash
PYTHONPATH=src python3 scripts/extract_audio_features.py \
  --input-manifest data/raw/train.jsonl \
  --output-manifest data/processed/train.jsonl \
  --feature-dir data/processed/audio_features
```

Extract cached video embeddings from raw video clips:

```bash
PYTHONPATH=src python3 scripts/extract_video_features.py \
  --input-manifest data/processed/train.jsonl \
  --output-manifest data/processed/train.jsonl \
  --feature-dir data/processed/video_features
```

## Notes

- The default project configuration is fully English-only.
- The text branch is explicitly standard BERT, not BERTurk.
- Audio and video extraction require the ML dependencies in `requirements.txt`.
- The demo UI downloads pretrained models on first use, so the first analysis run is slower than later runs.
- The current desktop environment used for authoring did not have `torch`, `transformers`, or `torchvision` installed, so the included smoke checks focus on schema, metrics, and synthetic data generation.
