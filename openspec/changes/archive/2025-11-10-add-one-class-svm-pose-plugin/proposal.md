# Change: Add One-Class SVM Pose Classification Plugin

## Why
The project currently has anomaly detection plugins using SVM and autoencoder for general pose anomalies, but lacks a dedicated one-class classification plugin that focuses on classifying poses into a single normal class with feature extraction. This would provide an additional method for pose classification in scenarios where only normal pose data is available for training.

## What Changes
- Add a new plugin that implements one-class SVM classification for pose data
- Include feature extraction from pose keypoints (e.g., angles, distances, ratios)
- Integrate with the existing plugin system and anomaly-detection capability in main pipeline
- Provide configurable parameters for SVM and feature extraction
- Add training functionality for the SVM model as command in pyproject.toml with CSV as input
- Input file is CSV format containing pose data: each row is a pose in a list x1, z1, x2, y2, ...  format
- Modify existing save plugin to order keypoints in a consistent order compatible with the feature extraction

## Impact
- Affected specs: anomaly-detection
- Affected code: poseplay/lib/ (new plugin file), poseplay/plugins.py (registration), poseplay/main.py (main pipeline integration), poseplay/lib/keypoints_save_plugin.py
- Breaking changes only to save plugin