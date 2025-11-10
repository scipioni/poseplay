"""Isolation Forest anomaly detection for PosePlay."""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


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

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the Isolation Forest on keypoints data.

        Args:
            keypoints_data: List of keypoints arrays, each shape (34,)
        """
        if not keypoints_data:
            raise ValueError("No training data provided")

        # Convert to numpy array
        X = np.array(keypoints_data)
        print(
            f"Training Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features"
        )

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


def load_csv_data(csv_path: str) -> List[np.ndarray]:
    """
    Load keypoints data from CSV file.

    Args:
        csv_path: Path to CSV file with keypoints in x1,y1,x2,y2,... format

    Returns:
        List of keypoints arrays
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    keypoints_list = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, 1):
            if len(row) != 34:  # 17 keypoints * 2 coordinates
                print(
                    f"Warning: Skipping row {row_num} - expected 34 values, got {len(row)}"
                )
                continue

            try:
                keypoints = np.array([float(x) for x in row], dtype=np.float32)
                keypoints_list.append(keypoints)
            except ValueError as e:
                print(f"Warning: Skipping row {row_num} - parsing error: {e}")

    print(f"Loaded {len(keypoints_list)} samples from {csv_path}")
    return keypoints_list


def train_isolation_forest_model(
    csv_path: str,
    model_path: str = "models/isolation_forest_anomaly_detector.pkl",
    n_estimators: int = 100,
    contamination: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    Train Isolation Forest anomaly detection model from CSV data.

    Args:
        csv_path: Path to CSV training data
        model_path: Path to save trained model
        n_estimators: Number of base estimators in the ensemble
        contamination: Expected proportion of outliers in the data
        random_state: Random state for reproducibility
    """
    print(f"Loading training data from {csv_path}")

    # Load keypoints data
    keypoints_data = load_csv_data(csv_path)

    if len(keypoints_data) == 0:
        raise ValueError("No valid samples found in CSV file")

    # Train the model
    detector = IsolationForestAnomalyDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    detector.train(keypoints_data)

    # Save the model
    detector.save(model_path)

    # Test on training data
    print("\nTesting model on training data:")
    normal_count = 0
    anomaly_count = 0

    for keypoints in keypoints_data[:10]:  # Test on first 10 samples
        is_anomaly, score = detector.detect(keypoints)
        if is_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1
        print(f"  Sample: anomaly={is_anomaly}, score={score:.4f}")

    print(f"Results: {normal_count} normal, {anomaly_count} anomalies detected")


def main():
    """Command line interface for training Isolation Forest anomaly detector."""
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest anomaly detection model"
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to CSV training data file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of base estimators in the ensemble",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected proportion of outliers in the data",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    args = parser.parse_args()

    if not args.model_path:
        args.model_path = Path(args.csv).with_suffix(".pkl")

    # Train the model
    train_isolation_forest_model(
        csv_path=args.csv,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
