## ADDED Requirements
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