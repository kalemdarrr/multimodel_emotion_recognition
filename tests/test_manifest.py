from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multimodal_emotion.data.manifest import ManifestSample, load_manifest, validate_manifest, write_manifest


class ManifestTests(unittest.TestCase):
    def test_write_and_load_manifest(self) -> None:
        sample = ManifestSample(
            sample_id="sample_001",
            label="joy",
            text="I am happy.",
            group_id="dialogue_001",
            audio_features_path="audio.npy",
            video_features_path="video.npy",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.jsonl"
            write_manifest([sample], path)
            loaded = load_manifest(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].sample_id, "sample_001")

    def test_validate_manifest(self) -> None:
        samples = [
            ManifestSample(
                sample_id="sample_001",
                label="joy",
                text="I am happy.",
                group_id="dialogue_001",
                audio_features_path="audio.npy",
                video_features_path="video.npy",
            ),
            ManifestSample(
                sample_id="sample_002",
                label="neutral",
                text="This is okay.",
                group_id="dialogue_001",
                audio_features_path="audio.npy",
                video_features_path="video.npy",
            ),
        ]
        report = validate_manifest(samples, ["neutral", "joy"])
        self.assertEqual(report["num_samples"], 2)
        self.assertEqual(report["num_groups"], 1)


if __name__ == "__main__":
    unittest.main()
