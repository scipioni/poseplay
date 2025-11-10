"""Autoencoder anomaly detection plugin for PosePlay."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from poseplay.autoencoder import AutoencoderAnomalyDetector
from poseplay.plugins import Plugin, PluginMetadata

logger = logging.getLogger(__name__)


class AutoencoderAnomalyPlugin(Plugin):
    """Plugin that detects pose anomalies using an autoencoder."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        latent_dim: int = 16,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        reconstruction_threshold: Optional[float] = None,
    ):
        """
        Initialize the autoencoder anomaly detection plugin.

        Args:
            model_path: Path to pre-trained autoencoder model file
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            reconstruction_threshold: Threshold for anomaly detection
        """
        self.model_path = model_path
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reconstruction_threshold = reconstruction_threshold
        self.detector: Optional[AutoencoderAnomalyDetector] = None
        self.anomaly_count = 0

        self.initialize()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="autoencoder_anomaly_plugin",
            version="1.0.0",
            description="Plugin for detecting pose anomalies using autoencoder reconstruction error",
            capabilities=["anomaly_detector", "image_processor"],
        )

    def initialize(self) -> None:
        """Initialize the plugin by loading or creating the autoencoder detector."""
        try:
            self.detector = AutoencoderAnomalyDetector(
                model_path=self.model_path,
                latent_dim=self.latent_dim,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                batch_size=self.batch_size,
                reconstruction_threshold=self.reconstruction_threshold,
            )
            logger.info("Initialized autoencoder anomaly detection plugin")
            if self.model_path:
                logger.info(f"Loaded pre-trained model from {self.model_path}")
            else:
                logger.warning("No model path provided - detector not trained yet")
        except Exception as e:
            logger.error(f"Failed to initialize autoencoder anomaly plugin: {e}")
            self.detector = None

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.detector:
            self.detector = None
            logger.info("Cleaned up autoencoder anomaly plugin")

    def process_frame(
        self, frame: np.ndarray, poses: Optional[List] = None
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Process the frame to detect pose anomalies.

        Args:
            frame: Input frame as numpy array with shape (height, width, 3)
            poses: Optional list of pose dictionaries from pose detection

        Returns:
            Tuple of (processed_frame, anomaly_results)
            - processed_frame: Frame with anomaly visualizations
            - anomaly_results: List of anomaly detection results for each pose
        """
        if self.detector is None or not self.detector.is_trained:
            logger.warning(
                "Autoencoder detector not trained, skipping anomaly detection"
            )
            return frame, None

        if poses is None:
            logger.debug("No poses provided for anomaly detection")
            return frame, None

        anomaly_results = []

        for pose in poses:
            # Extract keypoints as flat array
            xy = np.array(pose["xy"])  # pose["xy"] should be the keypoints array
            if xy is None:
                continue

            # Detect anomaly
            is_anomaly, error = self.detector.detect(xy)

            # Store result
            result = {
                "bbox": pose["bbox"],
                "is_anomaly": is_anomaly,
                "anomaly_score": float(error),  # reconstruction error
            }
            anomaly_results.append(result)

            # Log anomaly detection
            if is_anomaly:
                self.anomaly_count += 1
                logger.warning(
                    f"Anomaly detected! Reconstruction error: {error:.6f}, Count: {self.anomaly_count}"
                )

            # Visualize anomaly on frame
            self._visualize_anomaly(frame, result)

        return frame, anomaly_results

    def _visualize_anomaly(self, frame: np.ndarray, result: dict) -> None:
        """
        Visualize anomaly detection result on the frame.

        Args:
            frame: Frame to draw on
            result: Anomaly detection result dictionary
        """
        bbox = result["bbox"]
        is_anomaly = result["is_anomaly"]
        error = result["anomaly_score"]

        x1, y1, x2, y2 = bbox

        # Choose color based on anomaly status
        if is_anomaly:
            color = (0, 0, 255)  # Red for anomalies
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for normal
            thickness = 2

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Add reconstruction error text
        text = f"Error: {error:.4f}"
        cv2.putText(
            frame,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    def get_anomaly_stats(self) -> dict:
        """
        Get anomaly detection statistics.

        Returns:
            Dictionary with anomaly statistics
        """
        return {
            "total_anomalies": self.anomaly_count,
            "is_trained": self.detector.is_trained if self.detector else False,
            "reconstruction_threshold": self.detector.reconstruction_threshold
            if self.detector
            else None,
        }

    def train_on_data(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the autoencoder on provided keypoints data.

        Args:
            keypoints_data: List of keypoints arrays for training
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return

        try:
            self.detector.train(keypoints_data)
            logger.info("Autoencoder training completed")
        except Exception as e:
            logger.error(f"Failed to train autoencoder: {e}")

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to file.

        Args:
            model_path: Path to save the model
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return

        try:
            self.detector.save(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
