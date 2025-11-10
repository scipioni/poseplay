# anomaly-detection Specification

## Purpose
The anomaly-detection capability provides plugins for detecting anomalous human poses in video streams. This includes both SVM-based and autoencoder-based approaches for real-time fall detection and unusual pose identification.
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

### Requirement: Autoencoder Anomaly Detection Plugin
The system SHALL provide a plugin that uses an autoencoder neural network to detect anomalous human poses in real-time video streams by measuring reconstruction error.

#### Scenario: Normal Pose Reconstruction
- **WHEN** a pose with keypoints matching the trained normal distribution is detected
- **THEN** the plugin SHALL reconstruct the pose with low error and classify it as normal

#### Scenario: Anomalous Pose Reconstruction
- **WHEN** a pose with keypoints deviating from the trained normal distribution is detected
- **THEN** the plugin SHALL fail to reconstruct the pose accurately and classify it as anomalous based on high reconstruction error

#### Scenario: Autoencoder Model Training
- **WHEN** a dataset of normal pose keypoints is provided
- **THEN** the plugin SHALL train an autoencoder model to learn the normal pose manifold

#### Scenario: Model Persistence
- **WHEN** training is complete
- **THEN** the plugin SHALL save the trained autoencoder model and reconstruction threshold for later use

#### Scenario: Configurable Architecture
- **WHEN** latent dimension, learning rate, and training epochs are specified
- **THEN** the plugin SHALL use these parameters to configure the autoencoder architecture and training process

#### Scenario: Real-time Anomaly Detection
- **WHEN** processing video frames with pose keypoints
- **THEN** the plugin SHALL compute reconstruction error and provide anomaly detection results within frame processing time

#### Scenario: CSV Training Data Loading
- **WHEN** a CSV file containing pose keypoints data is provided
- **THEN** the plugin SHALL load the data where each row represents a pose with keypoints in x1,y1,x2,y2,... format

#### Scenario: Training Command Execution
- **WHEN** the training command is executed with CSV input file path
- **THEN** the plugin SHALL train the autoencoder model and save it to the specified output path

### Requirement: One-Class SVM Pose Classification Plugin
The system SHALL provide a plugin that uses One-Class SVM with feature extraction to classify human poses into a single normal class for anomaly detection in real-time video streams.

#### Scenario: Feature Extraction from Pose Keypoints
- **WHEN** pose keypoints are provided in COCO format (17 keypoints)
- **THEN** the plugin SHALL extract features including joint angles, limb lengths, and body ratios
- **ACCEPTANCE CRITERIA**: Features include shoulder-elbow-wrist angles, hip-knee-ankle angles, limb length ratios, and center of mass calculations; all features normalized and scaled appropriately

#### Scenario: One-Class SVM Training from CSV
- **WHEN** a CSV file containing normal pose keypoints is provided (each row is a pose in format x1,y1,x2,y2,... for 17 COCO keypoints)
- **THEN** the plugin SHALL parse the CSV, extract features from keypoints, and train a One-Class SVM model to learn the normal pose distribution
- **ACCEPTANCE CRITERIA**: Uses scikit-learn OneClassSVM with configurable nu, kernel, and gamma parameters; supports model persistence to disk; handles CSV parsing and feature extraction during training

#### Scenario: Real-time Pose Classification
- **WHEN** processing video frames with pose keypoints
- **THEN** the plugin SHALL extract features and classify poses as normal (anomaly_score < 0) or anomalous (anomaly_score >= 0)
- **ACCEPTANCE CRITERIA**: Classification performed within frame processing time; anomaly scores range from -1 (normal) to +1 (anomalous)

#### Scenario: Model Loading and Persistence
- **WHEN** a pre-trained One-Class SVM model path is provided
- **THEN** the plugin SHALL load the model and feature scaler for immediate classification
- **ACCEPTANCE CRITERIA**: Supports loading from .pkl files; handles missing model files gracefully with logging

#### Scenario: Configurable Parameters
- **WHEN** SVM parameters (nu, kernel, gamma) and feature extraction settings are specified
- **THEN** the plugin SHALL use these parameters for model configuration and feature processing
- **ACCEPTANCE CRITERIA**: Default values provided (nu=0.1, kernel='rbf', gamma='scale'); parameters configurable via plugin config

#### Scenario: Anomaly Visualization
- **WHEN** an anomalous pose is detected
- **THEN** the plugin SHALL overlay visual indicators on the frame (e.g., colored bounding boxes or text labels)
- **ACCEPTANCE CRITERIA**: Normal poses shown in green, anomalous in red; includes anomaly score in overlay text

#### Scenario: Training Command Integration
- **WHEN** the training command is invoked via pyproject.toml with CSV input
- **THEN** the plugin SHALL execute training using the provided CSV file and save the trained model
- **ACCEPTANCE CRITERIA**: Command accepts CSV file path and output model path as arguments; provides progress logging during training

