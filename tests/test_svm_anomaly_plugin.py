"""Tests for SVM anomaly detection plugin."""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from poseplay.lib.svm_anomaly_plugin import SVMAnomalyPlugin


class TestSVMAnomalyPlugin(unittest.TestCase):
    """Test cases for SVMAnomalyPlugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_keypoints = np.random.rand(34).astype(np.float32)
        self.test_pose = {
            'bbox': [10, 10, 100, 100],
            'keypoints': {
                'nose': {'x': 0.5, 'y': 0.2, 'confidence': 0.9},
                'left_shoulder': {'x': 0.4, 'y': 0.3, 'confidence': 0.8},
                'right_shoulder': {'x': 0.6, 'y': 0.3, 'confidence': 0.8},
                # Add more keypoints as needed for testing
            },
            'confidence': 0.95
        }

    def test_plugin_initialization_without_model(self):
        """Test plugin initialization without a pre-trained model."""
        plugin = SVMAnomalyPlugin()
        self.assertEqual(plugin.metadata.name, "svm_anomaly_plugin")
        self.assertEqual(plugin.metadata.version, "1.0.0")
        self.assertIn("anomaly_detector", plugin.metadata.capabilities)
        self.assertIsNotNone(plugin.detector)
        self.assertFalse(plugin.detector.is_trained)

    def test_plugin_initialization_with_model(self):
        """Test plugin initialization with a pre-trained model."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        try:
            # Create a mock trained detector
            plugin = SVMAnomalyPlugin(model_path=model_path)
            # Since no actual model exists, detector should be None or not trained
            if plugin.detector is not None:
                self.assertFalse(plugin.detector.is_trained)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_process_frame_without_poses(self):
        """Test processing frame without pose data."""
        plugin = SVMAnomalyPlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result_frame, results = plugin.process_frame(frame)
        np.testing.assert_array_equal(result_frame, frame)
        self.assertIsNone(results)

    def test_process_frame_without_trained_detector(self):
        """Test processing frame with untrained detector."""
        plugin = SVMAnomalyPlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [self.test_pose]

        result_frame, results = plugin.process_frame(frame, poses)
        np.testing.assert_array_equal(result_frame, frame)
        self.assertIsNone(results)

    @patch('poseplay.lib.svm_anomaly_plugin.OneClassSVMAnomalyDetector')
    def test_process_frame_with_trained_detector(self, mock_detector_class):
        """Test processing frame with trained detector."""
        # Mock the detector
        mock_detector = MagicMock()
        mock_detector.is_trained = True
        mock_detector.detect.return_value = (False, -0.5)  # Normal pose
        mock_detector_class.return_value = mock_detector

        plugin = SVMAnomalyPlugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [self.test_pose]

        result_frame, results = plugin.process_frame(frame, poses)

        # Check that detector was called
        mock_detector.detect.assert_called_once()
        # Check results structure
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertIn('is_anomaly', results[0])
        self.assertIn('anomaly_score', results[0])
        self.assertFalse(results[0]['is_anomaly'])

    @patch('poseplay.lib.svm_anomaly_plugin.OneClassSVMAnomalyDetector')
    def test_anomaly_detection_and_visualization(self, mock_detector_class):
        """Test anomaly detection and frame visualization."""
        # Mock detector that detects anomaly
        mock_detector = MagicMock()
        mock_detector.is_trained = True
        mock_detector.detect.return_value = (True, 0.8)  # Anomalous pose
        mock_detector_class.return_value = mock_detector

        plugin = SVMAnomalyPlugin()
        # Override the detector with our mock
        plugin.detector = mock_detector

        # Create a frame with some variation to ensure visualization changes it
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame
        poses = [self.test_pose]

        result_frame, results = plugin.process_frame(frame, poses)

        # Check that frame was modified (visualization added) - just check if any pixel changed
        # Since we're drawing on a black frame, any non-black pixels indicate drawing occurred
        has_drawing = np.any(result_frame != 0)
        self.assertTrue(has_drawing, "Frame should have been modified with anomaly visualization")
        # Check anomaly was detected
        self.assertTrue(results[0]['is_anomaly'])
        self.assertEqual(results[0]['anomaly_score'], 0.8)

    def test_keypoint_extraction(self):
        """Test keypoint extraction from pose dictionary."""
        plugin = SVMAnomalyPlugin()

        keypoints = plugin._extract_keypoints_array(self.test_pose)
        self.assertIsNotNone(keypoints)
        self.assertEqual(len(keypoints), 34)  # 17 keypoints * 2 coordinates

    def test_keypoint_extraction_missing_keypoints(self):
        """Test keypoint extraction with missing keypoints."""
        plugin = SVMAnomalyPlugin()
        incomplete_pose = {'bbox': [0, 0, 100, 100], 'keypoints': {}}

        keypoints = plugin._extract_keypoints_array(incomplete_pose)
        self.assertIsNotNone(keypoints)
        self.assertEqual(len(keypoints), 34)
        # Should be padded with zeros
        self.assertTrue(np.all(keypoints == 0.0))

    def test_get_anomaly_stats(self):
        """Test getting anomaly statistics."""
        plugin = SVMAnomalyPlugin()
        stats = plugin.get_anomaly_stats()

        self.assertIn('total_anomalies', stats)
        self.assertIn('is_trained', stats)
        self.assertEqual(stats['total_anomalies'], 0)
        self.assertFalse(stats['is_trained'])

    def test_cleanup(self):
        """Test plugin cleanup."""
        plugin = SVMAnomalyPlugin()
        plugin.cleanup()
        self.assertIsNone(plugin.detector)

    def test_configurable_parameters(self):
        """Test configurable SVM parameters."""
        plugin = SVMAnomalyPlugin(
            nu=0.2,
            kernel="linear",
            gamma="auto",
            anomaly_threshold=0.5
        )

        self.assertEqual(plugin.nu, 0.2)
        self.assertEqual(plugin.kernel, "linear")
        self.assertEqual(plugin.gamma, "auto")
        self.assertEqual(plugin.anomaly_threshold, 0.5)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()