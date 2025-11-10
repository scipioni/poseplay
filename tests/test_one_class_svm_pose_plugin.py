"""Tests for One-Class SVM Pose Classification Plugin."""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from poseplay.lib.one_class_svm_pose_plugin import (
    OneClassSVMPosePlugin,
    PoseFeatureExtractor,
)


class TestPoseFeatureExtractor(unittest.TestCase):
    """Test cases for PoseFeatureExtractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PoseFeatureExtractor()
        # Create sample keypoints for a standing pose
        self.sample_keypoints = np.array(
            [
                0.5,
                0.1,  # nose
                0.45,
                0.15,  # left_eye
                0.55,
                0.15,  # right_eye
                0.42,
                0.18,  # left_ear
                0.58,
                0.18,  # right_ear
                0.4,
                0.25,  # left_shoulder
                0.6,
                0.25,  # right_shoulder
                0.35,
                0.35,  # left_elbow
                0.65,
                0.35,  # right_elbow
                0.3,
                0.45,  # left_wrist
                0.7,
                0.45,  # right_wrist
                0.42,
                0.5,  # left_hip
                0.58,
                0.5,  # right_hip
                0.4,
                0.7,  # left_knee
                0.6,
                0.7,  # right_knee
                0.38,
                0.9,  # left_ankle
                0.62,
                0.9,  # right_ankle
            ],
            dtype=np.float32,
        )

    def test_calculate_angle(self):
        """Test angle calculation."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])

        angle = self.extractor.calculate_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_calculate_distance(self):
        """Test distance calculation."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])

        distance = self.extractor.calculate_distance(p1, p2)
        self.assertAlmostEqual(distance, 5.0, places=5)

    def test_extract_features(self):
        """Test feature extraction from keypoints."""
        features = self.extractor.extract_features(self.sample_keypoints)

        # Should extract some features
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_extract_features_invalid_shape(self):
        """Test feature extraction with invalid keypoint shape."""
        with self.assertRaises(ValueError):
            self.extractor.extract_features(np.array([1, 2, 3]))  # Wrong shape


class TestOneClassSVMPosePlugin(unittest.TestCase):
    """Test cases for OneClassSVMPosePlugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_keypoints = np.random.rand(34).astype(np.float32)
        self.test_pose = {
            "bbox": [10, 10, 100, 100],
            "xy": self.test_keypoints.tolist(),
            "confidence": 0.95,
        }

    def test_plugin_initialization_without_model(self):
        """Test plugin initialization without a pre-trained model."""
        plugin = OneClassSVMPosePlugin()
        self.assertEqual(plugin.metadata.name, "one_class_svm_pose_plugin")
        self.assertEqual(plugin.metadata.version, "1.0.0")
        self.assertIn("anomaly_detector", plugin.metadata.capabilities)
        self.assertIsNotNone(plugin.model)
        self.assertFalse(plugin.is_trained)

    def test_plugin_initialization_with_model(self):
        """Test plugin initialization with a pre-trained model."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            plugin = OneClassSVMPosePlugin(model_path=model_path)
            # Since no actual model exists, should not be trained
            self.assertFalse(plugin.is_trained)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_process_frame_without_poses(self):
        """Test processing frame without pose data."""
        plugin = OneClassSVMPosePlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result_frame, results = plugin.process_frame(frame)
        np.testing.assert_array_equal(result_frame, frame)
        self.assertIsNone(results)

    def test_process_frame_without_trained_model(self):
        """Test processing frame with untrained model."""
        plugin = OneClassSVMPosePlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [self.test_pose]

        result_frame, results = plugin.process_frame(frame, poses)
        np.testing.assert_array_equal(result_frame, frame)
        self.assertIsNone(results)

    @patch("sklearn.svm.OneClassSVM")
    @patch("sklearn.preprocessing.StandardScaler")
    def test_process_frame_with_trained_model(self, mock_scaler_class, mock_svm_class):
        """Test processing frame with trained model."""
        # Mock the SVM
        mock_svm = MagicMock()
        mock_svm.predict.return_value = [1]  # Normal pose
        mock_svm.decision_function.return_value = [-0.5]
        mock_svm_class.return_value = mock_svm

        # Mock the scaler
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]  # Mock scaled features
        mock_scaler_class.return_value = mock_scaler

        plugin = OneClassSVMPosePlugin()
        plugin.is_trained = True
        plugin.model = mock_svm
        plugin.scaler = mock_scaler

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [self.test_pose]

        result_frame, results = plugin.process_frame(frame, poses)

        # Check that SVM and scaler were called
        mock_scaler.transform.assert_called_once()
        mock_svm.predict.assert_called_once()
        mock_svm.decision_function.assert_called_once()

        # Check results structure
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertIn("is_anomaly", results[0])
        self.assertIn("anomaly_score", results[0])
        self.assertFalse(results[0]["is_anomaly"])

    def test_train(self):
        """Test training the model."""
        plugin = OneClassSVMPosePlugin()
        keypoints_data = [self.test_keypoints, self.test_keypoints * 0.9]

        plugin.train(keypoints_data)

        self.assertTrue(plugin.is_trained)
        self.assertIsNotNone(plugin.model)
        self.assertIsNotNone(plugin.scaler)

    def test_classify(self):
        """Test pose classification."""
        plugin = OneClassSVMPosePlugin()
        # Train with some data first
        keypoints_data = [self.test_keypoints]
        plugin.train(keypoints_data)

        # Extract features and classify
        features = plugin.feature_extractor.extract_features(self.test_keypoints)
        is_anomaly, score = plugin.classify(features)

        self.assertIsInstance(bool(is_anomaly), bool)
        self.assertIsInstance(score, float)

    def test_save_and_load_model(self):
        """Test saving and loading model."""
        plugin = OneClassSVMPosePlugin()
        keypoints_data = [self.test_keypoints]
        plugin.train(keypoints_data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            # Save model
            plugin.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Load model
            new_plugin = OneClassSVMPosePlugin()
            new_plugin.load_model(model_path)

            self.assertTrue(new_plugin.is_trained)
            self.assertIsNotNone(new_plugin.model)
            self.assertIsNotNone(new_plugin.scaler)

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_get_stats(self):
        """Test getting statistics."""
        plugin = OneClassSVMPosePlugin()
        stats = plugin.get_stats()

        self.assertIn("total_anomalies", stats)
        self.assertIn("is_trained", stats)
        self.assertEqual(stats["total_anomalies"], 0)
        self.assertFalse(stats["is_trained"])

    def test_cleanup(self):
        """Test plugin cleanup."""
        plugin = OneClassSVMPosePlugin()
        plugin.cleanup()
        self.assertIsNone(plugin.model)
        self.assertIsNone(plugin.scaler)

    def test_configurable_parameters(self):
        """Test configurable SVM parameters."""
        plugin = OneClassSVMPosePlugin(nu=0.2, kernel="linear", gamma="auto")

        self.assertEqual(plugin.nu, 0.2)
        self.assertEqual(plugin.kernel, "linear")
        self.assertEqual(plugin.gamma, "auto")


if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
