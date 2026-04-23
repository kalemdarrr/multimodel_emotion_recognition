# multimodel_emotion_recognition

English-only multimodal emotion recognition project built around the report blueprint. The system uses standard BERT for text, speech emotion analysis for audio, visual emotion analysis for video, and confidence-weighted late fusion for the final prediction.

## Overview

This repository implements a practical multimodal emotion recognition pipeline:

- Text branch: BERT on raw English utterances
- Audio branch: cached speech embeddings and speech-emotion scoring
- Video branch: frame sampling, face-centered visual emotion scoring
- Fusion branch: late fusion with robustness to weak or missing modalities
- Demo app: local upload interface for video emotion analysis

## What Is Included

- Configurable training pipeline for multimodal classification
- Manifest validation and grouped split utilities
- Synthetic dataset generation for smoke testing
- Audio feature extraction with Wav2Vec2-style encoders
- Video feature extraction with EfficientNet-based frame pooling
- Evaluation utilities for macro-F1, weighted-F1, accuracy, calibration, and confusion matrices
- Prediction and ONNX export scripts
- Local upload-based demo UI for video emotion analysis

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
  test_demo_fusion.py
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

Each sample is stored as one JSON object per line:

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

Training expects `text`, `label`, `audio_features_path`, and `video_features_path`.

## Quick Start

Generate a synthetic dataset:

```bash
PYTHONPATH=src python3 scripts/build_synthetic_dataset.py --output-dir data/synthetic_demo
```

Validate a manifest:

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

Train the model:

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

The demo uses the report-aligned first-stage setup:

- English ASR with Whisper
- Text emotion scoring with BERT
- Speech emotion scoring with Wav2Vec2
- Visual emotion scoring from sampled face frames
- Confidence-weighted late fusion across available modalities

## Notes

- The text branch is standard BERT, not BERTurk.
- The demo downloads pretrained models on first use, so the first run is slower.
- Generated artifacts, demo media, checkpoints, and local environments are ignored in git.
