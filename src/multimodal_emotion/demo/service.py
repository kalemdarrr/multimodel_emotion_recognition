from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from .fusion import (
    AUDIO_LABEL_MAP,
    TEXT_LABEL_MAP,
    VIDEO_LABEL_MAP,
    COMMON_LABELS,
    ModalitySummary,
    confidence_from_scores,
    remap_predictions,
    weighted_fusion,
)


@dataclass(slots=True)
class DemoModelConfig:
    asr_model_name: str = "openai/whisper-tiny.en"
    text_model_name: str = "bhadresh-savani/bert-base-uncased-emotion"
    audio_model_name: str = "superb/wav2vec2-base-superb-er"
    video_model_name: str = "RickyIG/emotion_face_image_classification"
    sample_frames: int = 8


class MultimodalDemoAnalyzer:
    def __init__(self, config: DemoModelConfig | None = None) -> None:
        self.config = config or DemoModelConfig()
        self.device = 0 if torch.cuda.is_available() else -1
        self.face_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        self._asr_pipeline = None
        self._text_pipeline = None
        self._audio_pipeline = None
        self._video_pipeline = None

    def preload_models(self) -> dict[str, str]:
        _ = self.asr_pipeline
        _ = self.text_pipeline
        _ = self.audio_pipeline
        _ = self.video_pipeline
        return {
            "asr": self.config.asr_model_name,
            "text": self.config.text_model_name,
            "audio": self.config.audio_model_name,
            "video": self.config.video_model_name,
        }

    @property
    def asr_pipeline(self):
        if self._asr_pipeline is None:
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.config.asr_model_name,
                device=self.device,
                chunk_length_s=20,
            )
        return self._asr_pipeline

    @property
    def text_pipeline(self):
        if self._text_pipeline is None:
            self._text_pipeline = pipeline(
                "text-classification",
                model=self.config.text_model_name,
                device=self.device,
                top_k=None,
            )
        return self._text_pipeline

    @property
    def audio_pipeline(self):
        if self._audio_pipeline is None:
            self._audio_pipeline = pipeline(
                "audio-classification",
                model=self.config.audio_model_name,
                device=self.device,
                top_k=None,
            )
        return self._audio_pipeline

    @property
    def video_pipeline(self):
        if self._video_pipeline is None:
            self._video_pipeline = pipeline(
                "image-classification",
                model=self.config.video_model_name,
                device=self.device,
                top_k=None,
            )
        return self._video_pipeline

    def extract_audio_track(self, video_path: str, output_dir: Path) -> tuple[Path | None, str]:
        audio_path = output_dir / "uploaded_audio.wav"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not audio_path.exists():
            return None, "No audio track detected. The app will continue with the available modalities."
        return audio_path, "Audio track extracted successfully."

    def transcribe_audio(self, audio_path: Path | None, transcript_override: str | None) -> tuple[str, str, str]:
        if transcript_override and transcript_override.strip():
            return transcript_override.strip(), "manual", "Manual transcript override used."
        if audio_path is None:
            return "", "missing", "No transcript available because the video had no readable audio track."
        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            return "", "missing", "ASR skipped transcription because the extracted waveform was empty."

        transcription = self.asr_pipeline(
            {"array": waveform.astype(np.float32), "sampling_rate": sample_rate}
        )
        transcript = str(transcription.get("text", "")).strip()
        if not transcript:
            return "", "missing", "ASR did not return speech content."
        return transcript, "whisper", "Transcript generated with Whisper ASR."

    def classify_text(self, transcript: str) -> ModalitySummary:
        if not transcript.strip():
            return ModalitySummary(
                name="text",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="No text transcript was available for the BERT branch.",
            )

        predictions = self.text_pipeline(transcript)
        prediction_list = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
        probabilities = remap_predictions(prediction_list, TEXT_LABEL_MAP)
        confidence = confidence_from_scores(probabilities)
        quality = min(max(len(transcript.split()) / 14.0, 0.35), 1.0)
        return ModalitySummary(
            name="text",
            status="ok",
            probabilities=probabilities,
            confidence=confidence,
            quality=quality,
            note="Text branch scored the English transcript with a BERT emotion classifier.",
        )

    def classify_audio(self, audio_path: Path | None) -> ModalitySummary:
        if audio_path is None:
            return ModalitySummary(
                name="audio",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="No audio track was available for the speech branch.",
            )

        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            return ModalitySummary(
                name="audio",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="Audio extraction succeeded but the waveform was empty.",
            )

        predictions = self.audio_pipeline({"array": waveform, "sampling_rate": sample_rate})
        prediction_list = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
        probabilities = remap_predictions(prediction_list, AUDIO_LABEL_MAP)
        confidence = confidence_from_scores(probabilities)
        duration_seconds = waveform.shape[0] / float(sample_rate)
        quality = min(max(duration_seconds / 6.0, 0.30), 1.0)
        return ModalitySummary(
            name="audio",
            status="ok",
            probabilities=probabilities,
            confidence=confidence,
            quality=quality,
            note="Audio branch scored vocal emotion from the extracted speech track.",
        )

    def _sample_frames(self, video_path: str) -> list[np.ndarray]:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames: list[np.ndarray] = []

        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num=min(self.config.sample_frames, total_frames), dtype=int)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = capture.read()
                if success:
                    sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            while len(sampled_frames) < self.config.sample_frames:
                success, frame = capture.read()
                if not success:
                    break
                sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        capture.release()
        return sampled_frames

    def _crop_face(self, frame: np.ndarray) -> tuple[Image.Image, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        if len(faces) > 0:
            x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
            crop = frame[y : y + height, x : x + width]
            return Image.fromarray(crop), True

        height, width = frame.shape[:2]
        side = int(min(height, width) * 0.70)
        start_y = max((height - side) // 2, 0)
        start_x = max((width - side) // 2, 0)
        crop = frame[start_y : start_y + side, start_x : start_x + side]
        return Image.fromarray(crop), False

    def classify_video(self, video_path: str) -> ModalitySummary:
        frames = self._sample_frames(video_path)
        if not frames:
            return ModalitySummary(
                name="video",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="No readable frames were found in the uploaded video.",
            )

        crops: list[Image.Image] = []
        detected_faces = 0
        for frame in frames:
            crop, face_found = self._crop_face(frame)
            detected_faces += int(face_found)
            crops.append(crop)

        predictions = self.video_pipeline(crops)
        if predictions and not isinstance(predictions[0], list):
            predictions = [predictions]

        aggregated = np.zeros(len(COMMON_LABELS), dtype=np.float64)
        for frame_predictions in predictions:
            remapped = remap_predictions(frame_predictions, VIDEO_LABEL_MAP)
            aggregated += np.array([remapped[label] for label in COMMON_LABELS], dtype=np.float64)
        aggregated = aggregated / aggregated.sum()
        probabilities = {label: float(aggregated[index]) for index, label in enumerate(COMMON_LABELS)}

        face_ratio = detected_faces / max(len(crops), 1)
        status = "ok" if detected_faces > 0 else "fallback"
        note = (
            "Video branch scored detected face crops sampled across the clip."
            if detected_faces > 0
            else "No faces were detected, so the video branch used centered frame crops as a fallback."
        )
        quality = max(0.30, face_ratio if detected_faces > 0 else 0.35)
        return ModalitySummary(
            name="video",
            status=status,
            probabilities=probabilities,
            confidence=confidence_from_scores(probabilities),
            quality=quality,
            note=note,
        )

    def analyze(self, video_path: str, transcript_override: str | None = None) -> dict:
        with tempfile.TemporaryDirectory(prefix="mme_demo_") as temp_dir:
            workspace = Path(temp_dir)
            audio_path, audio_note = self.extract_audio_track(video_path, workspace)
            transcript, transcript_source, transcript_note = self.transcribe_audio(audio_path, transcript_override)

            text_result = self.classify_text(transcript)
            audio_result = self.classify_audio(audio_path)
            video_result = self.classify_video(video_path)

            modalities = [text_result, audio_result, video_result]
            fusion = weighted_fusion(modalities)
            probability_rows = [
                {
                    "emotion": label.title(),
                    "score": round(fusion["probabilities"][label], 4),
                }
                for label in COMMON_LABELS
            ]
            modality_rows = [
                {
                    "modality": modality.name.title(),
                    "status": modality.status,
                    "top_emotion": max(modality.probabilities, key=modality.probabilities.get).title()
                    if any(modality.probabilities.values())
                    else "Unavailable",
                    "confidence": round(modality.confidence, 4),
                    "quality": round(modality.quality, 4),
                    "fusion_weight": round(fusion["modality_weights"].get(modality.name, 0.0), 4),
                    "note": modality.note,
                }
                for modality in modalities
            ]
            return {
                "predicted_label": fusion["predicted_label"],
                "confidence": fusion["confidence"],
                "probabilities": fusion["probabilities"],
                "probability_rows": probability_rows,
                "modality_rows": modality_rows,
                "transcript": transcript,
                "transcript_source": transcript_source,
                "notes": {
                    "audio": audio_note,
                    "transcript": transcript_note,
                },
            }
