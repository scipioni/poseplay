"""Tests for Isolation Forest anomaly detection plugin."""

import numpy as np
import pytest

from poseplay.lib.isolation_forest_anomaly_plugin import IsolationForestAnomalyPlugin


class TestIsolationForestAnomalyPlugin:
    """Test cases for the Isolation Forest anomaly detection plugin."""

    def test_plugin_initialization(self):
        """Test plugin initialization without model path."""
        plugin = IsolationForestAnomalyPlugin()
        assert plugin.metadata.name == "isolation_forest_anomaly_plugin"
        assert plugin.metadata.version == "1.0.0"
        assert "anomaly_detector" in plugin.metadata.capabilities
        assert "image_processor" in plugin.metadata.capabilities
        assert plugin.detector is not None
        assert not plugin.detector.is_trained

    def test_plugin_initialization_with_params(self):
        """Test plugin initialization with custom parameters."""
        plugin = IsolationForestAnomalyPlugin(
            n_estimators=50,
            contamination=0.2,
            random_state=123
        )
        assert plugin.n_estimators == 50
        assert plugin.contamination == 0.2
        assert plugin.random_state == 123

    def test_process_frame_without_training(self):
        """Test processing frame when detector is not trained."""
        plugin = IsolationForestAnomalyPlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [{"xy": np.random.rand(34), "bbox": [10, 10, 50, 50]}]

        result_frame, results = plugin.process_frame(frame, poses)
        assert results is None  # Should return None when not trained
        np.testing.assert_array_equal(result_frame, frame)  # Frame unchanged

    def test_process_frame_without_poses(self):
        """Test processing frame without poses."""
        plugin = IsolationForestAnomalyPlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result_frame, results = plugin.process_frame(frame, None)
        assert results is None
        np.testing.assert_array_equal(result_frame, frame)

    def test_training_and_detection(self):
        """Test training the detector and performing anomaly detection."""
        plugin = IsolationForestAnomalyPlugin(n_estimators=10, random_state=42)

        # Generate training data (normal poses)
        np.random.seed(42)
        training_data = [np.random.normal(0, 1, 34) for _ in range(50)]

        # Train the plugin
        plugin.train_on_data(training_data)
        assert plugin.detector.is_trained

        # Test detection on normal data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        normal_pose = {"xy": np.random.normal(0, 1, 34), "bbox": [10, 10, 50, 50]}
        poses = [normal_pose]

        result_frame, results = plugin.process_frame(frame, poses)
        assert results is not None
        assert len(results) == 1
        assert "is_anomaly" in results[0]
        assert "anomaly_score" in results[0]
        assert "bbox" in results[0]

        # Test detection on anomalous data (far from training distribution)
        anomalous_pose = {"xy": np.random.normal(10, 1, 34), "bbox": [60, 60, 100, 100]}
        poses = [anomalous_pose]

        result_frame, results = plugin.process_frame(frame, poses)
        assert results is not None
        assert len(results) == 1

    def test_visualization(self):
        """Test that visualization modifies the frame."""
        plugin = IsolationForestAnomalyPlugin(n_estimators=10, random_state=42)

        # Train with some data
        training_data = [np.random.normal(0, 1, 34) for _ in range(20)]
        plugin.train_on_data(training_data)

        # Create test frame and pose
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pose = {"xy": np.random.normal(0, 1, 34), "bbox": [10, 10, 50, 50]}

        # Process frame
        result_frame, _ = plugin.process_frame(frame, [pose])

        # Frame should be modified (visualization added)
        # Note: We can't easily test exact pixel values, but frame should be different
        assert result_frame.shape == frame.shape

    def test_get_anomaly_stats(self):
        """Test getting anomaly statistics."""
        plugin = IsolationForestAnomalyPlugin()
        stats = plugin.get_anomaly_stats()

        expected_keys = {"total_anomalies", "is_trained", "contamination"}
        assert set(stats.keys()) == expected_keys
        assert stats["total_anomalies"] == 0
        assert not stats["is_trained"]
        assert stats["contamination"] == 0.1

    def test_cleanup(self):
        """Test plugin cleanup."""
        plugin = IsolationForestAnomalyPlugin()
        assert plugin.detector is not None

        plugin.cleanup()
        assert plugin.detector is None

    def test_save_model(self, tmp_path):
        """Test saving the trained model."""
        plugin = IsolationForestAnomalyPlugin(n_estimators=10, random_state=42)

        # Train the model
        training_data = [np.random.normal(0, 1, 34) for _ in range(20)]
        plugin.train_on_data(training_data)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        plugin.save_model(str(model_path))
        assert model_path.exists()

    def test_invalid_keypoints_shape(self):
        """Test handling of invalid keypoints shape."""
        plugin = IsolationForestAnomalyPlugin(n_estimators=10, random_state=42)

        # Train the model
        training_data = [np.random.normal(0, 1, 34) for _ in range(20)]
        plugin.train_on_data(training_data)

        # Test with wrong shape
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pose = {"xy": np.random.normal(0, 1, 17), "bbox": [10, 10, 50, 50]}  # Only 17 values

        # Should handle gracefully (skip invalid poses)
        result_frame, results = plugin.process_frame(frame, [pose])
        assert results is not None
        assert len(results) == 0  # Invalid pose should be skipped