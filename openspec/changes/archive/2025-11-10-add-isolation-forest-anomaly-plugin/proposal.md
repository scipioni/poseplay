# Change: Add Isolation Forest Anomaly Detection Plugin

## Why
The project currently has SVM and autoencoder-based anomaly detection plugins, but lacks an Isolation Forest implementation. Isolation Forest is a powerful unsupervised anomaly detection method that works well for high-dimensional data like pose keypoints and is computationally efficient compared to other methods.

## What Changes
- Add a new isolation_forest_anomaly_plugin.py in the poseplay/lib/ directory
- Add training functionality for the autoencoder model as command in pyproject.toml with CSV as input
- Input file is CSV format containing pose data: each row is a pose in a list x1, z1, x2, y2, ...  format
- Implement Isolation Forest anomaly detection for pose keypoints
- Add plugin metadata and capabilities for anomaly detection
- Include visualization of anomaly scores on frames
- Support configurable contamination parameter and random state

## Impact
- Affected specs: anomaly-detection (adds new plugin capability)
- Affected code: new plugin file in poseplay/lib/ directory, poseplay/main.py
- New capability: isolation Forest anomaly detection alongside existing methods
- No breaking changes to existing plugin interfaces