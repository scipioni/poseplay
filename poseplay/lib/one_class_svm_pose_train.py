"""Training script for One-Class SVM Pose Classification Plugin."""

import argparse
import csv
import os
from pathlib import Path
from typing import List

import numpy as np

from .one_class_svm_pose_plugin import OneClassSVMPosePlugin


def load_csv_data(csv_path: str) -> List[np.ndarray]:
    """
    Load keypoints data from CSV file.

    Args:
        csv_path: Path to CSV file containing pose keypoints

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

    print(f"Loaded {len(keypoints_list)} pose samples from {csv_path}")
    return keypoints_list


def train_one_class_svm_pose_model(
    csv_path: str,
    model_path: str = "models/one_class_svm_pose_model.pkl",
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> None:
    """
    Train One-Class SVM pose classification model from CSV data.

    Args:
        csv_path: Path to CSV training data
        model_path: Path to save trained model
        nu: Anomaly parameter
        kernel: Kernel type
        gamma: Kernel coefficient
    """
    print(f"Loading training data from {csv_path}")

    # Load keypoints data
    keypoints_data = load_csv_data(csv_path)

    if len(keypoints_data) == 0:
        raise ValueError(f"No valid pose data found in {csv_path}")

    # Create and train the plugin
    plugin = OneClassSVMPosePlugin(nu=nu, kernel=kernel, gamma=gamma)
    plugin.train(keypoints_data)

    # Save the model
    plugin.save_model(model_path)

    # Test on training data
    print("\nTesting model on training data:")
    normal_count = 0
    anomaly_count = 0

    for keypoints in keypoints_data[:10]:  # Test on first 10 samples
        try:
            features = plugin.feature_extractor.extract_features(keypoints)
            is_anomaly, score = plugin.classify(features)
            if is_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1
            print(f"  Sample: anomaly={is_anomaly}, score={score:.4f}")
        except Exception as e:
            print(f"  Sample failed: {e}")

    print(f"Results: {normal_count} normal, {anomaly_count} anomalies detected")


def main():
    """Command line interface for training One-Class SVM pose model."""
    parser = argparse.ArgumentParser(
        description="Train One-Class SVM pose classification model"
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
        "--nu", type=float, default=0.1, help="Anomaly parameter (0 < nu <= 1)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="Kernel type",
    )
    parser.add_argument("--gamma", type=str, default="scale", help="Kernel coefficient")

    args = parser.parse_args()

    # Validate arguments
    if not 0 < args.nu <= 1:
        parser.error("--nu must be between 0 and 1")

    if not args.model_path:
        args.model_path = Path(args.csv).with_suffix(".pkl")

    # Train the model
    train_one_class_svm_pose_model(
        csv_path=args.csv,
        model_path=args.model_path,
        nu=args.nu,
        kernel=args.kernel,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
