"""Isolation Forest anomaly detection plugin for PosePlay."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from poseplay.plugins import Plugin, PluginMetadata

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """
    Isolation Forest anomaly detector for pose keypoints.
    Uses scikit-learn's IsolationForest for unsupervised anomaly detection.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the Isolation Forest anomaly detector.

        Args:
            model_path: Path to saved model file
            n_estimators: Number of base estimators in the ensemble
            contamination: Expected proportion of outliers in the data
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.is_trained = False

        if model_path:
            self.load(model_path)

    def train(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the Isolation Forest on keypoints data.

        Args:
            keypoints_data: List of keypoints arrays, each shape (34,)
        """
        if not keypoints_data:
            raise ValueError("No training data provided")

        from sklearn.ensemble import IsolationForest

        # Convert to numpy array
        X = np.array(keypoints_data)
        print(f"Training Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")

        # Initialize and train the model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self.is_trained = True

        print(
            f"Isolation Forest training completed. "
            f"n_estimators={self.n_estimators}, contamination={self.contamination}"
        )

    def detect(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if keypoints are anomalous.

        Args:
            keypoints: Keypoints array, shape (34,)

        Returns:
            Tuple of (is_anomaly, anomaly_score)
            - is_anomaly: True if anomalous, False if normal
            - anomaly_score: Anomaly score (negative for normal, positive for anomalies)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before detection")

        # Ensure correct shape
        if keypoints.shape != (34,):
            keypoints = keypoints.flatten()
            if keypoints.shape != (34,):
                raise ValueError(f"Expected 34 keypoints, got {keypoints.shape}")

        # Predict anomaly score
        score = self.model.decision_function([keypoints])[0]
        prediction = self.model.predict([keypoints])[0]  # 1 for normal, -1 for anomaly

        is_anomaly = prediction == -1
        return is_anomaly, float(score)

    def save(self, model_path: str) -> None:
        """
        Save the trained model to file.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        import os
        import pickle

        os.makedirs(
            os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
            exist_ok=True,
        )

        model_data = {
            "model": self.model,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "is_trained": self.is_trained,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Isolation Forest model saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load a trained model from file.

        Args:
            model_path: Path to the saved model
        """
        import os
        import pickle

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.n_estimators = model_data["n_estimators"]
        self.contamination = model_data["contamination"]
        self.random_state = model_data["random_state"]
        self.is_trained = model_data["is_trained"]

        print(f"Isolation Forest model loaded from {model_path}")


class IsolationForestAnomalyPlugin(Plugin):
    """Plugin that detects pose anomalies using Isolation Forest."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the Isolation Forest anomaly detection plugin.

        Args:
            model_path: Path to pre-trained Isolation Forest model file
            n_estimators: Number of base estimators in the ensemble
            contamination: Expected proportion of outliers in the data
            random_state: Random state for reproducibility
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.detector: Optional[IsolationForestAnomalyDetector] = None
        self.anomaly_count = 0

        self.initialize()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="isolation_forest_anomaly_plugin",
            version="1.0.0",
            description="Plugin for detecting pose anomalies using Isolation Forest",
            capabilities=["anomaly_detector", "image_processor"]
        )

    def initialize(self) -> None:
        """Initialize the plugin by loading or creating the Isolation Forest detector."""
        try:
            self.detector = IsolationForestAnomalyDetector(
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
            )
            logger.info("Initialized Isolation Forest anomaly detection plugin")
            if self.model_path:
                logger.info(f"Loaded pre-trained model from {self.model_path}")
            else:
                logger.warning("No model path provided - detector not trained yet")
        except Exception as e:
            logger.error(f"Failed to initialize Isolation Forest anomaly plugin: {e}")
            self.detector = None

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.detector:
            self.detector = None
            logger.info("Cleaned up Isolation Forest anomaly plugin")

    def process_frame(self, frame: np.ndarray, poses: Optional[List] = None) -> Tuple[np.ndarray, Optional[List]]:
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
            logger.warning("Isolation Forest detector not trained, skipping anomaly detection")
            return frame, None

        if poses is None:
            logger.debug("No poses provided for anomaly detection")
            return frame, None

        anomaly_results = []

        for pose in poses:
            # Extract keypoints as flat array
            xy = np.array(pose["xy"])  # pose["xy"] should be the keypoints array
            if xy is None or xy.size != 34:
                continue

            # Detect anomaly
            is_anomaly, score = self.detector.detect(xy)

            # Store result
            result = {
                'bbox': pose['bbox'],
                'is_anomaly': is_anomaly,
                'anomaly_score': float(score),  # anomaly score
            }
            anomaly_results.append(result)

            # Log anomaly detection
            if is_anomaly:
                self.anomaly_count += 1
                logger.warning(f"Anomaly detected! Score: {score:.4f}, Count: {self.anomaly_count}")

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
        bbox = result['bbox']
        is_anomaly = result['is_anomaly']
        score = result['anomaly_score']

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

        # Add anomaly score text
        text = f"Score: {score:.2f}"
        cv2.putText(frame, text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_anomaly_stats(self) -> dict:
        """
        Get anomaly detection statistics.

        Returns:
            Dictionary with anomaly statistics
        """
        return {
            'total_anomalies': self.anomaly_count,
            'is_trained': self.detector.is_trained if self.detector else False,
            'contamination': self.contamination,
        }

    def train_on_data(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the Isolation Forest on provided keypoints data.

        Args:
            keypoints_data: List of keypoints arrays for training
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return

        try:
            self.detector.train(keypoints_data)
            logger.info("Isolation Forest training completed")
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest: {e}")

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