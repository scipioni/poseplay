# anomaly-detection Specification

## Purpose
TBD - created by archiving change add-svm-anomaly-plugin. Update Purpose after archive.
## Requirements
### Requirement: SVM Anomaly Detection Plugin
The system SHALL provide a plugin that uses One-Class SVM to detect anomalous human poses in real-time video streams.

#### Scenario: Normal Pose Detection
- **WHEN** a pose with keypoints matching the trained normal distribution is detected
- **THEN** the plugin SHALL classify it as normal and return anomaly_score < 0

#### Scenario: Anomalous Pose Detection
- **WHEN** a pose with keypoints deviating from the trained normal distribution is detected
- **THEN** the plugin SHALL classify it as anomalous and return anomaly_score >= 0

#### Scenario: Model Loading
- **WHEN** a pre-trained SVM model path is provided
- **THEN** the plugin SHALL load the model and scaler for immediate anomaly detection

#### Scenario: Real-time Processing
- **WHEN** processing video frames with pose keypoints
- **THEN** the plugin SHALL analyze keypoints and provide anomaly detection results within frame processing time

#### Scenario: Configurable Parameters
- **WHEN** nu, kernel, and gamma parameters are specified
- **THEN** the plugin SHALL use these parameters for SVM model configuration

#### Scenario: Anomaly Visualization
- **WHEN** an anomalous pose is detected
- **THEN** the plugin SHALL overlay visual indicators (e.g., colored bounding boxes) on the frame

