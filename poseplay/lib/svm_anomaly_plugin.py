"""SVM anomaly detection plugin for PosePlay."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from poseplay.plugins import Plugin, PluginMetadata
from poseplay.svm import OneClassSVMAnomalyDetector

logger = logging.getLogger(__name__)


class SVMAnomalyPlugin(Plugin):
    """Plugin that detects pose anomalies using One-Class SVM."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
        anomaly_threshold: float = 0.0,
    ):
        """
        Initialize the SVM anomaly detection plugin.

        Args:
            model_path: Path to pre-trained SVM model file
            nu: Anomaly parameter for One-Class SVM (0 < nu <= 1)
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
            anomaly_threshold: Threshold for anomaly score (scores >= threshold are anomalous)
        """
        self.model_path = model_path
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.anomaly_threshold = anomaly_threshold
        self.detector: Optional[OneClassSVMAnomalyDetector] = None
        self.anomaly_count = 0

        self.initialize()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="svm_anomaly_plugin",
            version="1.0.0",
            description="Plugin for detecting pose anomalies using One-Class SVM",
            capabilities=["anomaly_detector", "image_processor"],
        )

    def initialize(self) -> None:
        """Initialize the plugin by loading or creating the SVM detector."""
        try:
            self.detector = OneClassSVMAnomalyDetector(
                model_path=self.model_path,
                nu=self.nu,
                kernel=self.kernel,
                gamma=self.gamma,
            )
            logger.info("Initialized SVM anomaly detection plugin")
            if self.model_path:
                logger.info(f"Loaded pre-trained model from {self.model_path}")
            else:
                logger.warning("No model path provided - detector not trained yet")
        except Exception as e:
            logger.error(f"Failed to initialize SVM anomaly plugin: {e}")
            self.detector = None

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.detector:
            self.detector = None
            logger.info("Cleaned up SVM anomaly plugin")

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
            logger.warning("SVM detector not trained, skipping anomaly detection")
            return frame, None

        if poses is None:
            logger.debug("No poses provided for anomaly detection")
            return frame, None

        anomaly_results = []

        for pose in poses:
            # Extract keypoints as flat array
            xy = np.array(pose["xy"])  # self._extract_keypoints_array(pose)
            if xy is None:
                continue

            # Detect anomaly
            is_anomaly, score = self.detector.detect(xy)

            # Store result
            result = {
                "bbox": pose["bbox"],
                "is_anomaly": is_anomaly,
                "anomaly_score": float(score),
                #'keypoints': keypoints
            }
            anomaly_results.append(result)

            # Log anomaly detection
            if is_anomaly:
                self.anomaly_count += 1
                logger.warning(
                    f"Anomaly detected! Score: {score:.4f}, Count: {self.anomaly_count}"
                )

            # Visualize anomaly on frame
            self._visualize_anomaly(frame, result)

        return frame, anomaly_results

    # def _extract_keypoints_array(self, pose: dict) -> Optional[np.ndarray]:
    #     """
    #     Extract keypoints from pose dictionary as flat array.

    #     Args:
    #         pose: Pose dictionary with keypoints

    #     Returns:
    #         Flattened keypoints array (34,) or None if extraction fails
    #     """
    #     try:
    #         keypoints_dict = pose.get('keypoints', {})

    #         # Extract x,y coordinates for all 17 keypoints in order
    #         keypoints = []
    #         for i in range(17):  # 17 COCO keypoints
    #             keypoint_name = self._get_keypoint_name(i)
    #             if keypoint_name in keypoints_dict:
    #                 kp = keypoints_dict[keypoint_name]
    #                 keypoints.extend([kp['x'], kp['y']])
    #             else:
    #                 # Pad with zeros if keypoint missing
    #                 keypoints.extend([0.0, 0.0])

    #         return np.array(keypoints, dtype=np.float32)

    #     except Exception as e:
    #         logger.warning(f"Failed to extract keypoints from pose: {e}")
    #         return None

    # def _get_keypoint_name(self, index: int) -> str:
    #     """Get keypoint name by index."""
    #     keypoint_names = [
    #         "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    #         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    #         "left_wrist", "right_wrist", "left_hip", "right_hip",
    #         "left_knee", "right_knee", "left_ankle", "right_ankle"
    #     ]
    #     return keypoint_names[index] if 0 <= index < len(keypoint_names) else f"keypoint_{index}"

    def _visualize_anomaly(self, frame: np.ndarray, result: dict) -> None:
        """
        Visualize anomaly detection result on the frame.

        Args:
            frame: Frame to draw on
            result: Anomaly detection result dictionary
        """
        bbox = result["bbox"]
        is_anomaly = result["is_anomaly"]
        score = result["anomaly_score"]

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
        text = f"Anomaly: {score:.2f}"
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
        }
