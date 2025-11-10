"""One-Class SVM Pose Classification Plugin with feature extraction."""

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from poseplay.plugins import Plugin, PluginMetadata

logger = logging.getLogger(__name__)


class PoseFeatureExtractor:
    """Extract features from pose keypoints for anomaly detection."""

    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by points p1, p2, p3."""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle floating point errors
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    @staticmethod
    def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return float(np.linalg.norm(p1 - p2))

    def extract_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract features from pose keypoints.

        Args:
            keypoints: Array of shape (34,) with x,y coordinates for 17 keypoints

        Returns:
            Feature vector
        """
        if keypoints.shape != (34,):
            raise ValueError(f"Expected 34 keypoints, got {keypoints.shape}")

        # Reshape to (17, 2) for easier indexing
        kp = keypoints.reshape(17, 2)

        features = []

        # Joint angles
        # Shoulder-elbow-wrist angles
        if self._is_valid_triangle(
            kp, self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST
        ):
            angle = self.calculate_angle(
                kp[self.LEFT_SHOULDER], kp[self.LEFT_ELBOW], kp[self.LEFT_WRIST]
            )
            features.append(angle)

        if self._is_valid_triangle(
            kp, self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST
        ):
            angle = self.calculate_angle(
                kp[self.RIGHT_SHOULDER], kp[self.RIGHT_ELBOW], kp[self.RIGHT_WRIST]
            )
            features.append(angle)

        # Hip-knee-ankle angles
        if self._is_valid_triangle(kp, self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE):
            angle = self.calculate_angle(
                kp[self.LEFT_HIP], kp[self.LEFT_KNEE], kp[self.LEFT_ANKLE]
            )
            features.append(angle)

        if self._is_valid_triangle(
            kp, self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE
        ):
            angle = self.calculate_angle(
                kp[self.RIGHT_HIP], kp[self.RIGHT_KNEE], kp[self.RIGHT_ANKLE]
            )
            features.append(angle)

        # Elbow angles (shoulder-elbow-wrist)
        if self._is_valid_triangle(
            kp, self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST
        ):
            angle = self.calculate_angle(
                kp[self.LEFT_SHOULDER], kp[self.LEFT_ELBOW], kp[self.LEFT_WRIST]
            )
            features.append(angle)

        if self._is_valid_triangle(
            kp, self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST
        ):
            angle = self.calculate_angle(
                kp[self.RIGHT_SHOULDER], kp[self.RIGHT_ELBOW], kp[self.RIGHT_WRIST]
            )
            features.append(angle)

        # Limb lengths
        # Upper limbs
        left_upper_arm = self.calculate_distance(
            kp[self.LEFT_SHOULDER], kp[self.LEFT_ELBOW]
        )
        right_upper_arm = self.calculate_distance(
            kp[self.RIGHT_SHOULDER], kp[self.RIGHT_ELBOW]
        )
        left_forearm = self.calculate_distance(kp[self.LEFT_ELBOW], kp[self.LEFT_WRIST])
        right_forearm = self.calculate_distance(
            kp[self.RIGHT_ELBOW], kp[self.RIGHT_WRIST]
        )

        features.extend([left_upper_arm, right_upper_arm, left_forearm, right_forearm])

        # Lower limbs
        left_thigh = self.calculate_distance(kp[self.LEFT_HIP], kp[self.LEFT_KNEE])
        right_thigh = self.calculate_distance(kp[self.RIGHT_HIP], kp[self.RIGHT_KNEE])
        left_shin = self.calculate_distance(kp[self.LEFT_KNEE], kp[self.LEFT_ANKLE])
        right_shin = self.calculate_distance(kp[self.RIGHT_KNEE], kp[self.RIGHT_ANKLE])

        features.extend([left_thigh, right_thigh, left_shin, right_shin])

        # Body ratios
        # Shoulder width
        shoulder_width = self.calculate_distance(
            kp[self.LEFT_SHOULDER], kp[self.RIGHT_SHOULDER]
        )
        features.append(shoulder_width)

        # Hip width
        hip_width = self.calculate_distance(kp[self.LEFT_HIP], kp[self.RIGHT_HIP])
        features.append(hip_width)

        # Torso height (shoulder to hip center)
        left_shoulder_hip = (kp[self.LEFT_SHOULDER] + kp[self.LEFT_HIP]) / 2
        right_shoulder_hip = (kp[self.RIGHT_SHOULDER] + kp[self.RIGHT_HIP]) / 2
        torso_height = (
            self.calculate_distance(kp[self.LEFT_SHOULDER], left_shoulder_hip)
            + self.calculate_distance(kp[self.RIGHT_SHOULDER], right_shoulder_hip)
        ) / 2
        features.append(torso_height)

        # Center of mass approximation (weighted average of key points)
        # Using major joints with weights
        weights = np.array(
            [
                1,
                1,
                1,
                1,  # head
                2,
                2,
                2,
                2,
                2,
                2,  # arms
                3,
                3,
                3,
                3,
                3,
                3,
            ]
        )  # legs (higher weight)

        # Calculate weighted center
        weighted_sum = np.zeros(2)
        total_weight = 0

        for i, w in enumerate(weights):
            if i < len(kp):
                weighted_sum += kp[i] * w
                total_weight += w

        if total_weight > 0:
            com = weighted_sum / total_weight
            # Distance from nose to COM
            nose_to_com = self.calculate_distance(kp[self.NOSE], com)
            features.append(nose_to_com)

        # Symmetry features (ratios between left and right)
        if left_upper_arm > 0 and right_upper_arm > 0:
            arm_ratio = left_upper_arm / right_upper_arm
            features.append(arm_ratio)

        if left_thigh > 0 and right_thigh > 0:
            leg_ratio = left_thigh / right_thigh
            features.append(leg_ratio)

        # Ensure we have at least some features
        if not features:
            features = [0.0]  # Fallback feature

        return np.array(features, dtype=np.float32)

    def _is_valid_triangle(
        self, keypoints: np.ndarray, i1: int, i2: int, i3: int
    ) -> bool:
        """Check if three keypoints form a valid triangle (not collinear)."""
        if i1 >= len(keypoints) or i2 >= len(keypoints) or i3 >= len(keypoints):
            return False

        p1, p2, p3 = keypoints[i1], keypoints[i2], keypoints[i3]

        # Check if points are not too close (degenerate triangle)
        d1 = self.calculate_distance(p1, p2)
        d2 = self.calculate_distance(p2, p3)
        d3 = self.calculate_distance(p1, p3)

        return d1 > 1e-6 and d2 > 1e-6 and d3 > 1e-6


class OneClassSVMPosePlugin(Plugin):
    """Plugin that classifies poses using One-Class SVM with feature extraction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
    ):
        """
        Initialize the One-Class SVM pose classification plugin.

        Args:
            model_path: Path to pre-trained SVM model file
            nu: Anomaly parameter for One-Class SVM (0 < nu <= 1)
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
        """
        self.model_path = model_path
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.scaler = None
        self.feature_extractor = PoseFeatureExtractor()
        self.is_trained = False
        self.anomaly_count = 0

        self.initialize()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="one_class_svm_pose_plugin",
            version="1.0.0",
            description="Plugin for pose classification using One-Class SVM with feature extraction",
            capabilities=["anomaly_detector", "image_processor"],
        )

    def initialize(self) -> None:
        """Initialize the plugin by loading or creating the SVM model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.load_model(self.model_path)
                logger.info(f"Loaded pre-trained model from {self.model_path}")
            else:
                from sklearn.svm import OneClassSVM

                self.model = OneClassSVM(
                    nu=self.nu, kernel=self.kernel, gamma=self.gamma
                )
                self.scaler = StandardScaler()
                logger.info("Initialized One-Class SVM pose plugin")
                if not self.model_path:
                    logger.warning("No model path provided - model not trained yet")
        except Exception as e:
            logger.error(f"Failed to initialize One-Class SVM pose plugin: {e}")
            self.model = None
            self.scaler = None

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.model:
            self.model = None
            self.scaler = None
            logger.info("Cleaned up One-Class SVM pose plugin")

    def process_frame(
        self, frame: np.ndarray, poses: Optional[List] = None
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Process the frame to classify poses.

        Args:
            frame: Input frame as numpy array with shape (height, width, 3)
            poses: Optional list of pose dictionaries from pose detection

        Returns:
            Tuple of (processed_frame, anomaly_results)
            - processed_frame: Frame with anomaly visualizations
            - anomaly_results: List of classification results for each pose
        """
        if self.model is None or not self.is_trained:
            logger.warning("SVM model not trained, skipping pose classification")
            return frame, None

        if poses is None:
            logger.debug("No poses provided for classification")
            return frame, None

        anomaly_results = []

        for pose in poses:
            # Extract keypoints as flat array
            xy = np.array(pose["xy"]).flatten()

            # Extract features
            try:
                features = self.feature_extractor.extract_features(xy)
            except Exception as e:
                logger.warning(f"Failed to extract features: {e}")
                continue

            # Classify pose
            is_anomaly, score = self.classify(features)

            # Store result
            result = {
                "bbox": pose["bbox"],
                "is_anomaly": is_anomaly,
                "anomaly_score": float(score),
            }
            anomaly_results.append(result)

            # Log classification
            if is_anomaly:
                self.anomaly_count += 1
                logger.warning(
                    f"Anomalous pose detected! Score: {score:.4f}, Count: {self.anomaly_count}"
                )

            # Visualize result on frame
            self._visualize_result(frame, result)

        return frame, anomaly_results

    def classify(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Classify pose features as normal or anomalous.

        Args:
            features: Feature vector

        Returns:
            Tuple of (is_anomaly, anomaly_score)
            - is_anomaly: True if anomalous, False if normal
            - anomaly_score: Negative for normal, positive for anomalous
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before classification")

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Predict
        prediction = self.model.predict(features_scaled)[
            0
        ]  # 1 for normal, -1 for anomaly
        score = self.model.decision_function(features_scaled)[0]

        is_anomaly = prediction == -1
        return is_anomaly, float(score)

    def train(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the One-Class SVM on normal pose keypoints.

        Args:
            keypoints_data: List of keypoints arrays, each shape (34,)
        """
        if not keypoints_data:
            raise ValueError("No training data provided")

        print(f"Training One-Class SVM on {len(keypoints_data)} samples")

        # Extract features from all training samples
        features_list = []
        for keypoints in keypoints_data:
            try:
                features = self.feature_extractor.extract_features(keypoints)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to extract features from sample: {e}")
                continue

        if not features_list:
            raise ValueError("No valid features extracted from training data")

        X = np.array(features_list)
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train the model
        self.model.fit(X_scaled)
        self.is_trained = True

        print(
            f"One-Class SVM training completed. Nu={self.nu}, kernel={self.kernel}, gamma={self.gamma}"
        )

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model and scaler to file.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        import os

        os.makedirs(
            os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
            exist_ok=True,
        )

        import pickle

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_extractor": self.feature_extractor,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "is_trained": self.is_trained,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"One-Class SVM model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model and scaler from file.

        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        import pickle

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_extractor = model_data["feature_extractor"]
        self.nu = model_data["nu"]
        self.kernel = model_data["kernel"]
        self.gamma = model_data["gamma"]
        self.is_trained = model_data["is_trained"]

        print(f"One-Class SVM model loaded from {model_path}")

    def _visualize_result(self, frame: np.ndarray, result: dict) -> None:
        """
        Visualize classification result on the frame.

        Args:
            frame: Frame to draw on
            result: Classification result dictionary
        """
        bbox = result["bbox"]
        is_anomaly = result["is_anomaly"]
        score = result["anomaly_score"]

        x1, y1, x2, y2 = bbox

        # Choose color based on classification
        if is_anomaly:
            color = (0, 0, 255)  # Red for anomalies
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for normal
            thickness = 2

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Add score text
        text = f"Score: {score:.2f}"
        cv2.putText(
            frame,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    def get_stats(self) -> dict:
        """
        Get classification statistics.

        Returns:
            Dictionary with statistics
        """
        return {"total_anomalies": self.anomaly_count, "is_trained": self.is_trained}
