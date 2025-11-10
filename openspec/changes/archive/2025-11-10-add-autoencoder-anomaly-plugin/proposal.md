# Change: Add Autoencoder Anomaly Detection Plugin

## Why
The current SVM anomaly detection plugin is limited to one-class classification. Adding an autoencoder-based plugin will provide a more flexible anomaly detection approach using reconstruction error, potentially offering better performance for complex pose patterns.

## What Changes
- Add new autoencoder anomaly detection plugin in `poseplay/lib/autoencoder_anomaly_plugin.py`
- Implement one-class classifier using autoencoder for pose anomaly detection
- Add training functionality for the autoencoder model as command in pyproject.toml with CSV as input
- Input file is CSV format containing pose data: each row is a pose in a list x1, z1, x2, y2, ...  format
- Integrate with existing plugin system and pose detection pipeline
- Add configuration options for autoencoder parameters (latent dimension, learning rate, epochs)

## Impact
- Affected specs: anomaly-detection
- Affected code: poseplay/lib/, poseplay/plugins.py, poseplay/main.py
- New capability: autoencoder-based anomaly detection alongside existing SVM approach

## Acceptance Criteria
- Plugin can be loaded and initialized without errors
- Autoencoder can be trained on pose keypoints data
- Plugin detects anomalies in real-time video processing
- Reconstruction error threshold is configurable
- Model can be saved and loaded for persistence