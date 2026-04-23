from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COMMON_LABELS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

MODALITY_BASE_WEIGHTS = {
    "text": 0.40,
    "audio": 0.35,
    "video": 0.25,
}

TEXT_LABEL_MAP: dict[str, dict[str, float]] = {
    "anger": {"anger": 1.0},
    "fear": {"fear": 1.0},
    "joy": {"joy": 1.0},
    "love": {"joy": 0.85, "surprise": 0.15},
    "neutral": {"neutral": 1.0},
    "sadness": {"sadness": 1.0},
    "surprise": {"surprise": 1.0},
}

AUDIO_LABEL_MAP: dict[str, dict[str, float]] = {
    "ang": {"anger": 1.0},
    "anger": {"anger": 1.0},
    "hap": {"joy": 1.0},
    "happy": {"joy": 1.0},
    "joy": {"joy": 1.0},
    "neu": {"neutral": 1.0},
    "neutral": {"neutral": 1.0},
    "sad": {"sadness": 1.0},
    "sadness": {"sadness": 1.0},
}

VIDEO_LABEL_MAP: dict[str, dict[str, float]] = {
    "angry": {"anger": 1.0},
    "anger": {"anger": 1.0},
    "disgust": {"disgust": 1.0},
    "disgusted": {"disgust": 1.0},
    "fear": {"fear": 1.0},
    "fearful": {"fear": 1.0},
    "happy": {"joy": 1.0},
    "joy": {"joy": 1.0},
    "neutral": {"neutral": 1.0},
    "sad": {"sadness": 1.0},
    "sadness": {"sadness": 1.0},
    "surprise": {"surprise": 1.0},
    "surprised": {"surprise": 1.0},
}

LABEL_ALIASES = {
    "ang": "ang",
    "anger": "anger",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgusted",
    "fear": "fear",
    "fearful": "fearful",
    "hap": "hap",
    "happy": "happy",
    "joy": "joy",
    "love": "love",
    "neu": "neu",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprised",
}


@dataclass(slots=True)
class ModalitySummary:
    name: str
    status: str
    probabilities: dict[str, float]
    confidence: float
    quality: float
    note: str


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return LABEL_ALIASES.get(normalized, normalized)


def remap_predictions(
    predictions: list[dict],
    label_map: dict[str, dict[str, float]],
    *,
    smoothing: float = 1e-6,
) -> dict[str, float]:
    scores = np.full(len(COMMON_LABELS), smoothing, dtype=np.float64)
    label_to_index = {label: index for index, label in enumerate(COMMON_LABELS)}

    for prediction in predictions:
        raw_label = normalize_label(str(prediction["label"]))
        if raw_label not in label_map:
            continue
        for target_label, weight in label_map[raw_label].items():
            scores[label_to_index[target_label]] += float(prediction["score"]) * weight

    scores = scores / scores.sum()
    return {label: float(scores[index]) for index, label in enumerate(COMMON_LABELS)}


def confidence_from_scores(probabilities: dict[str, float]) -> float:
    return float(max(probabilities.values(), default=0.0))


def weighted_fusion(modalities: list[ModalitySummary]) -> dict:
    label_to_index = {label: index for index, label in enumerate(COMMON_LABELS)}
    fused = np.zeros(len(COMMON_LABELS), dtype=np.float64)
    modality_weights: dict[str, float] = {}

    for modality in modalities:
        if modality.status not in {"ok", "fallback"}:
            continue

        base_weight = MODALITY_BASE_WEIGHTS[modality.name]
        quality_weight = max(modality.quality, 0.20)
        confidence_weight = max(modality.confidence, 0.10)
        final_weight = base_weight * quality_weight * confidence_weight
        modality_weights[modality.name] = float(final_weight)

        for label, score in modality.probabilities.items():
            fused[label_to_index[label]] += final_weight * score

    if not modality_weights:
        raise ValueError("At least one modality must be available for fusion.")

    fused = fused / fused.sum()
    probabilities = {label: float(fused[index]) for index, label in enumerate(COMMON_LABELS)}
    predicted_label = max(probabilities, key=probabilities.get)
    return {
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "confidence": float(probabilities[predicted_label]),
        "modality_weights": modality_weights,
    }
