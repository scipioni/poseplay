"""Autoencoder anomaly detection for PosePlay."""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """Simple autoencoder for pose keypoints anomaly detection."""

    def __init__(self, input_dim: int = 34, latent_dim: int = 16):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Dimension of input keypoints (34 for 17 keypoints * 2 coords)
            latent_dim: Dimension of latent space
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(x)


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detector for pose keypoints.
    Trained on normal poses to learn reconstruction, detects anomalies by reconstruction error.
    """

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
        Initialize the autoencoder anomaly detector.

        Args:
            model_path: Path to saved model file
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            reconstruction_threshold: Threshold for anomaly detection (auto-calculated if None)
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reconstruction_threshold = reconstruction_threshold
        self.model = None
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _create_model(self) -> None:
        """Create the autoencoder model."""
        self.model = Autoencoder(latent_dim=self.latent_dim).to(self.device)

    def train(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the autoencoder on normal keypoints data.

        Args:
            keypoints_data: List of keypoints arrays, each shape (34,)
        """
        if not keypoints_data:
            raise ValueError("No training data provided")

        # Convert to numpy array
        X = np.array(keypoints_data)
        print(
            f"Training autoencoder on {X.shape[0]} samples with {X.shape[1]} features"
        )

        # Create model if not exists
        if self.model is None:
            self._create_model()

        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # Input and target are the same
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                # Forward pass
                output = self.model(batch_x)
                loss = criterion(output, batch_x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.6f}")

        # Calculate reconstruction threshold if not provided
        if self.reconstruction_threshold is None:
            self._calculate_threshold(X_tensor)

        self.is_trained = True
        print(f"Autoencoder training completed. Latent dim: {self.latent_dim}")

    def _calculate_threshold(self, X: torch.Tensor) -> None:
        """Calculate reconstruction threshold based on training data."""
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1)
            # Set threshold as mean + 2*std of reconstruction errors
            self.reconstruction_threshold = float(errors.mean() + 2 * errors.std())
        print(
            f"Calculated reconstruction threshold: {self.reconstruction_threshold:.6f}"
        )

    def detect(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if keypoints are anomalous based on reconstruction error.

        Args:
            keypoints: Keypoints array, shape (34,)

        Returns:
            Tuple of (is_anomaly, reconstruction_error)
            - is_anomaly: True if anomalous, False if normal
            - reconstruction_error: MSE reconstruction error
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before detection")

        # Ensure correct shape
        if keypoints.shape != (34,):
            keypoints = keypoints.flatten()
            if keypoints.shape != (34,):
                raise ValueError(f"Expected 34 keypoints, got {keypoints.shape}")

        # Convert to tensor
        keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)

        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(keypoints_tensor)
            error = torch.mean((keypoints_tensor - reconstructed) ** 2).item()

        is_anomaly = error > self.reconstruction_threshold
        return is_anomaly, float(error)

    def save(self, model_path: str) -> None:
        """
        Save the trained model and parameters to file.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        os.makedirs(
            os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
            exist_ok=True,
        )

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "reconstruction_threshold": self.reconstruction_threshold,
            "is_trained": self.is_trained,
        }

        torch.save(model_data, model_path)
        print(f"Autoencoder model saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load a trained model and parameters from file.

        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = torch.load(model_path, map_location=self.device)

        # Create model
        self._create_model()
        self.model.load_state_dict(model_data["model_state_dict"])

        # Load parameters
        self.latent_dim = model_data["latent_dim"]
        self.learning_rate = model_data["learning_rate"]
        self.epochs = model_data["epochs"]
        self.batch_size = model_data["batch_size"]
        self.reconstruction_threshold = model_data["reconstruction_threshold"]
        self.is_trained = model_data["is_trained"]

        print(f"Autoencoder model loaded from {model_path}")


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


def train_autoencoder_model(
    csv_path: str,
    model_path: str = "models/autoencoder_anomaly_detector.pth",
    latent_dim: int = 16,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
) -> None:
    """
    Train autoencoder anomaly detection model from CSV data.

    Args:
        csv_path: Path to CSV training data
        model_path: Path to save trained model
        latent_dim: Dimension of latent space
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print(f"Loading training data from {csv_path}")

    # Load keypoints data
    keypoints_data = load_csv_data(csv_path)

    if len(keypoints_data) == 0:
        raise ValueError("No valid samples found in CSV file")

    # Train the model
    detector = AutoencoderAnomalyDetector(
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )
    detector.train(keypoints_data)

    # Save the model
    detector.save(model_path)

    # Test on training data
    print("\nTesting model on training data:")
    normal_count = 0
    anomaly_count = 0

    for keypoints in keypoints_data[:10]:  # Test on first 10 samples
        is_anomaly, error = detector.detect(keypoints)
        if is_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1
        print(f"  Sample: anomaly={is_anomaly}, error={error:.6f}")

    print(f"Results: {normal_count} normal, {anomaly_count} anomalies detected")


def main():
    """Command line interface for training autoencoder anomaly detector."""
    parser = argparse.ArgumentParser(
        description="Train autoencoder anomaly detection model"
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
        "--latent-dim", type=int, default=16, help="Dimension of latent space"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )

    args = parser.parse_args()

    if not args.model_path:
        args.model_path = Path(args.csv).with_suffix(".pth")

    # Train the model
    train_autoencoder_model(
        csv_path=args.csv,
        model_path=args.model_path,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
