# PosePlay

A flexible pose detection application that supports various image input sources including files, directories, videos, and RTSP streams.


```
# create data/rooms/termica.csv
pose grab data/rooms/termica.mkv --save

# create data/rooms/termica.pkl (SVM model)
svmtrain --csv data/rooms/termica.csv --grid-search

# create data/rooms/termica.pth (Autoencoder model)
autoencodertrain --csv data/rooms/termica.csv --latent-dim 16 --epochs 100

# create data/rooms/termica_if.pkl (Isolation Forest model)
isolationforesttrain --csv data/rooms/termica.csv --n-estimators 100 --contamination 0.1

# detect anomalies with SVM
pose grab data/rooms/termica.mkv --svm data/rooms/termica.pkl

# detect anomalies with autoencoder
pose grab data/rooms/termica.mkv --autoencoder data/rooms/termica.pth

# detect anomalies with Isolation Forest
pose grab data/rooms/termica.mkv --isolation-forest data/rooms/termica_if.pkl
```



## Features

- **Multiple Input Sources**: Support for single images, image directories, video files, and RTSP streams
- **Unified Interface**: Consistent API across all input types
- **Real-time Processing**: Frame-by-frame processing with configurable frame rates
- **Plugin System**: Extensible plugin architecture for custom image processing
- **Pose Detection**: YOLO-based human pose estimation with keypoint detection
- **Anomaly Detection**: SVM-based, autoencoder-based, and Isolation Forest-based pose anomaly detection for fall detection and unusual pose identification
- **Keypoint Saving**: CSV export of detected pose keypoints for training and analysis
- **CLI Interface**: Command-line interface for easy integration
- **Keyboard Controls**: Interactive controls for pause, resume, and reset operations

## CSV Format for Training Data

The training data for anomaly detection models should be in CSV format with the following structure:

- **Format**: Each row represents a single pose
- **Columns**: 34 numeric values representing 17 keypoints (x1,y1,x2,y2,...,x17,y17)
- **Data Type**: Float values representing keypoint coordinates
- **Header**: No header row (data starts immediately)

Example CSV structure:
```
0.5,0.3,0.6,0.4,0.4,0.2,...  # First pose keypoints
0.7,0.5,0.8,0.6,0.5,0.3,...  # Second pose keypoints
...
```

This format is used by both SVM and autoencoder training commands.

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

Grab images from a single file:
```bash
python -m poseplay grab path/to/image.jpg
```

Grab images from a directory:
```bash
python -m poseplay grab path/to/image/directory/
```

Grab frames from a video file:
```bash
python -m poseplay grab path/to/video.mp4
```

Grab frames from an RTSP stream:
```bash
python -m poseplay grab rtsp://example.com/stream
```

### Options

- `--fps FLOAT`: Set frame rate for video processing (default: 30.0)
- `--max-retries INT`: Maximum retries for RTSP connection failures (default: 3)
- `--loop`: Loop continuously when source ends (restarts from beginning)
- `--plugins LIST`: Comma-separated list of plugins to load (e.g., yolo_pose_plugin,svm_anomaly_plugin)
- `--save`: Save pose keypoints to CSV file
- `--svm PATH`: Path to pre-trained SVM anomaly detection model
- `--autoencoder PATH`: Path to pre-trained autoencoder anomaly detection model
- `--isolation-forest PATH`: Path to pre-trained Isolation Forest anomaly detection model

### Plugins

PosePlay supports extensible plugins for various image processing tasks:

- **yolo_pose_plugin**: Detects human poses using YOLO and annotates frames with keypoints and skeleton
- **keypoints_save_plugin**: Saves detected pose keypoints to CSV files for training
- **svm_anomaly_plugin**: Uses One-Class SVM to detect anomalous poses in real-time
- **autoencoder_anomaly_plugin**: Uses autoencoder neural network to detect anomalous poses by reconstruction error
- **isolation_forest_anomaly_plugin**: Uses Isolation Forest algorithm to detect anomalous poses in high-dimensional keypoint data

### Anomaly Detection

PosePlay supports three anomaly detection approaches: SVM-based, autoencoder-based, and Isolation Forest-based.

#### SVM Anomaly Detection

Train an anomaly detection model from normal pose data:

```bash
# 1. Collect normal pose keypoints
python -m poseplay grab video.mp4 --save

# 2. Train SVM model on collected keypoints
python poseplay/svm.py --csv keypoints.csv --model-path anomaly_model.pkl --grid-search

# 3. Use trained model for real-time anomaly detection
python -m poseplay grab rtsp://camera/stream --svm anomaly_model.pkl
```

SVM Parameters:
- `--nu FLOAT`: Anomaly parameter (0 < nu <= 1, default: 0.1)
- `--kernel STR`: Kernel type (rbf, linear, poly, sigmoid, default: rbf)
- `--gamma STR`: Kernel coefficient (scale, auto, or float, default: scale)

#### Autoencoder Anomaly Detection

Train an autoencoder model for anomaly detection:

```bash
# 1. Collect normal pose keypoints (same as SVM)
python -m poseplay grab video.mp4 --save

# 2. Train autoencoder model on collected keypoints
python -m poseplay.autoencoder --csv keypoints.csv --model-path autoencoder_model.pth --latent-dim 16 --epochs 100

# 3. Use trained model for real-time anomaly detection
python -m poseplay grab rtsp://camera/stream --autoencoder autoencoder_model.pth
```

Autoencoder Parameters:
- `--latent-dim INT`: Dimension of latent space (default: 16)
- `--learning-rate FLOAT`: Learning rate for training (default: 0.001)
- `--epochs INT`: Number of training epochs (default: 100)
- `--batch-size INT`: Batch size for training (default: 32)

#### Isolation Forest Anomaly Detection

Train an Isolation Forest model for anomaly detection:

```bash
# 1. Collect normal pose keypoints (same as SVM/autoencoder)
python -m poseplay grab video.mp4 --save

# 2. Train Isolation Forest model on collected keypoints
python -m poseplay.isolation_forest --csv keypoints.csv --model-path isolation_forest_model.pkl --n-estimators 100 --contamination 0.1

# 3. Use trained model for real-time anomaly detection
python -m poseplay grab rtsp://camera/stream --isolation-forest isolation_forest_model.pkl
```

Isolation Forest Parameters:
- `--n-estimators INT`: Number of base estimators in the ensemble (default: 100)
- `--contamination FLOAT`: Expected proportion of outliers in the data (default: 0.1)
- `--random-state INT`: Random state for reproducibility (default: 42)

### Keyboard Controls

During playback (except RTSP streams):
- `q` or `ESC`: Quit the application
- `p` or `SPACE`: Pause/resume playback
- `r`: Reset the grabber (restart from beginning)


### Running Tests

```bash
python -m pytest tests/
```

### Project Structure

```
poseplay/
├── __init__.py
├── main.py              # CLI entry point
├── config.py            # Configuration handling
├── image_grabber.py     # Core image grabbing functionality
├── plugins.py           # Plugin system implementation
├── svm.py               # SVM anomaly detection utilities
├── autoencoder.py       # Autoencoder anomaly detection utilities
├── isolation_forest.py  # Isolation Forest anomaly detection utilities
└── lib/
    ├── yolo_pose_plugin.py      # YOLO pose detection plugin
    ├── keypoints_save_plugin.py # Keypoints CSV saving plugin
    ├── svm_anomaly_plugin.py    # SVM anomaly detection plugin
    ├── autoencoder_anomaly_plugin.py # Autoencoder anomaly detection plugin
    └── isolation_forest_anomaly_plugin.py # Isolation Forest anomaly detection plugin

docs/
└── plugin-development.md # Plugin development guide

tests/
├── test_image_grabber.py     # Unit tests for image grabber
├── test_plugins.py          # Unit tests for plugin system
├── test_svm_anomaly_plugin.py # Unit tests for SVM anomaly plugin
├── test_autoencoder_anomaly_plugin.py # Unit tests for autoencoder anomaly plugin
└── test_isolation_forest_anomaly_plugin.py # Unit tests for Isolation Forest anomaly plugin
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License.