"""Tests for autoencoder anomaly detection plugin."""

import numpy as np
import pytest
import torch

from poseplay.autoencoder import AutoencoderAnomalyDetector, load_csv_data
from poseplay.lib.autoencoder_anomaly_plugin import AutoencoderAnomalyPlugin


class TestAutoencoderAnomalyDetector:
    """Test the autoencoder anomaly detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = AutoencoderAnomalyDetector()
        assert not detector.is_trained
        assert detector.model is None

    def test_training(self):
        """Test training with sample data."""
        # Create sample training data
        keypoints_data = [
            np.random.rand(34).astype(np.float32) for _ in range(20)
        ]

        detector = AutoencoderAnomalyDetector(latent_dim=8, epochs=2)
        detector.train(keypoints_data)

        assert detector.is_trained
        assert detector.model is not None
        assert detector.reconstruction_threshold > 0

    def test_detection_normal(self):
        """Test anomaly detection on normal data."""
        # Create training data
        keypoints_data = [
            np.random.rand(34).astype(np.float32) for _ in range(20)
        ]

        detector = AutoencoderAnomalyDetector(latent_dim=8, epochs=2)
        detector.train(keypoints_data)

        # Test on similar data
        test_keypoints = np.random.rand(34).astype(np.float32)
        is_anomaly, error = detector.detect(test_keypoints)

        assert isinstance(is_anomaly, bool)
        assert isinstance(error, float)
        assert error >= 0

    def test_detection_untrained(self):
        """Test detection fails when untrained."""
        detector = AutoencoderAnomalyDetector()
        test_keypoints = np.random.rand(34).astype(np.float32)

        with pytest.raises(RuntimeError):
            detector.detect(test_keypoints)

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        # Train a model
        keypoints_data = [
            np.random.rand(34).astype(np.float32) for _ in range(20)
        ]
        detector = AutoencoderAnomalyDetector(latent_dim=8, epochs=2)
        detector.train(keypoints_data)

        # Save model
        model_path = tmp_path / "test_model.pth"
        detector.save(str(model_path))

        # Load model with same latent dimension
        new_detector = AutoencoderAnomalyDetector(
            model_path=str(model_path),
            latent_dim=8  # Must match saved model
        )
        assert new_detector.is_trained
        assert abs(new_detector.reconstruction_threshold - detector.reconstruction_threshold) < 1e-6

    def test_invalid_keypoints_shape(self):
        """Test handling of invalid keypoints shape."""
        detector = AutoencoderAnomalyDetector()
        detector._create_model()
        detector.is_trained = True

        # Test with wrong shape
        invalid_keypoints = np.random.rand(30)  # Should be 34

        with pytest.raises(ValueError):
            detector.detect(invalid_keypoints)


class TestLoadCsvData:
    """Test CSV data loading."""

    def test_load_valid_csv(self, tmp_path):
        """Test loading valid CSV data."""
        # Create test CSV
        csv_path = tmp_path / "test.csv"
        data = []
        for _ in range(5):
            row = [str(np.random.rand()) for _ in range(34)]
            data.append(row)

        with open(csv_path, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(data)

        # Load data
        keypoints = load_csv_data(str(csv_path))
        assert len(keypoints) == 5
        assert all(kp.shape == (34,) for kp in keypoints)

    def test_load_invalid_csv(self, tmp_path):
        """Test loading invalid CSV data."""
        csv_path = tmp_path / "invalid.csv"

        # Create CSV with wrong number of columns
        with open(csv_path, 'w', newline='') as f:
            f.write("1,2,3\n")  # Only 3 values, need 34

        keypoints = load_csv_data(str(csv_path))
        assert len(keypoints) == 0  # Should skip invalid rows

    def test_load_nonexistent_csv(self):
        """Test loading nonexistent CSV."""
        with pytest.raises(FileNotFoundError):
            load_csv_data("nonexistent.csv")


class TestAutoencoderAnomalyPlugin:
    """Test the autoencoder anomaly plugin."""

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        plugin = AutoencoderAnomalyPlugin()
        assert plugin.metadata.name == "autoencoder_anomaly_plugin"
        assert "anomaly_detector" in plugin.metadata.capabilities

    def test_plugin_process_frame_no_poses(self):
        """Test processing frame with no poses."""
        plugin = AutoencoderAnomalyPlugin()
        frame = np.random.rand(480, 640, 3).astype(np.uint8)

        result_frame, anomalies = plugin.process_frame(frame, None)
        assert anomalies is None
        np.testing.assert_array_equal(result_frame, frame)

    def test_plugin_process_frame_untrained(self):
        """Test processing frame when detector is untrained."""
        plugin = AutoencoderAnomalyPlugin()
        frame = np.random.rand(480, 640, 3).astype(np.uint8)
        poses = [{"xy": np.random.rand(34), "bbox": [10, 10, 50, 50]}]

        result_frame, anomalies = plugin.process_frame(frame, poses)
        assert anomalies is None
        np.testing.assert_array_equal(result_frame, frame)

    def test_plugin_process_frame_trained(self):
        """Test processing frame with trained detector."""
        # Create and train detector
        keypoints_data = [
            np.random.rand(34).astype(np.float32) for _ in range(20)
        ]
        plugin = AutoencoderAnomalyPlugin(latent_dim=8, epochs=2)
        plugin.train_on_data(keypoints_data)

        # Process frame
        frame = np.random.rand(480, 640, 3).astype(np.uint8)
        poses = [{"xy": np.random.rand(34), "bbox": [10, 10, 50, 50]}]

        result_frame, anomalies = plugin.process_frame(frame, poses)
        assert anomalies is not None
        assert len(anomalies) == 1
        assert "is_anomaly" in anomalies[0]
        assert "anomaly_score" in anomalies[0]

    def test_plugin_stats(self):
        """Test getting plugin statistics."""
        plugin = AutoencoderAnomalyPlugin()
        stats = plugin.get_anomaly_stats()

        assert "total_anomalies" in stats
        assert "is_trained" in stats
        assert "reconstruction_threshold" in stats