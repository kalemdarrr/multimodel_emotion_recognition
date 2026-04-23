from __future__ import annotations

import unittest

from multimodal_emotion.demo.fusion import (
    AUDIO_LABEL_MAP,
    TEXT_LABEL_MAP,
    ModalitySummary,
    normalize_label,
    remap_predictions,
    weighted_fusion,
)


class DemoFusionTests(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(normalize_label("ANG"), "ang")
        self.assertEqual(normalize_label("Happy"), "happy")

    def test_remap_predictions(self) -> None:
        predictions = [
            {"label": "love", "score": 0.8},
            {"label": "joy", "score": 0.2},
        ]
        scores = remap_predictions(predictions, TEXT_LABEL_MAP)
        self.assertGreater(scores["joy"], scores["surprise"])

    def test_weighted_fusion_prefers_supported_modalities(self) -> None:
        text = ModalitySummary(
            name="text",
            status="ok",
            probabilities=remap_predictions([{"label": "joy", "score": 1.0}], TEXT_LABEL_MAP),
            confidence=0.90,
            quality=1.0,
            note="",
        )
        audio = ModalitySummary(
            name="audio",
            status="ok",
            probabilities=remap_predictions([{"label": "hap", "score": 1.0}], AUDIO_LABEL_MAP),
            confidence=0.85,
            quality=0.8,
            note="",
        )
        result = weighted_fusion([text, audio])
        self.assertEqual(result["predicted_label"], "joy")


if __name__ == "__main__":
    unittest.main()
