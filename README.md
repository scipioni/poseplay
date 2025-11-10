# PosePlay

A flexible pose detection application that supports various image input sources including files, directories, videos, and RTSP streams.


```
# create data/rooms/termica.csv
pose grab data/rooms/termica.mkv --save 

# create data/rooms/termica.pkl
svmtrain --csv data/rooms/termica.csv --grid-search 

# detect anomalies
pose grab data/rooms/termica.mkv --svm data/rooms/termica.pkl
```



## Features

- **Multiple Input Sources**: Support for single images, image directories, video files, and RTSP streams
- **Unified Interface**: Consistent API across all input types
- **Real-time Processing**: Frame-by-frame processing with configurable frame rates
- **Plugin System**: Extensible plugin architecture for custom image processing
- **Pose Detection**: YOLO-based human pose estimation with keypoint detection
- **Anomaly Detection**: SVM-based pose anomaly detection for fall detection and unusual pose identification
- **Keypoint Saving**: CSV export of detected pose keypoints for training and analysis
- **CLI Interface**: Command-line interface for easy integration
- **Keyboard Controls**: Interactive controls for pause, resume, and reset operations

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
- `--model-path PATH`: Path to pre-trained SVM anomaly detection model

### Plugins

PosePlay supports extensible plugins for various image processing tasks:

- **yolo_pose_plugin**: Detects human poses using YOLO and annotates frames with keypoints and skeleton
- **keypoints_save_plugin**: Saves detected pose keypoints to CSV files for training
- **svm_anomaly_plugin**: Uses One-Class SVM to detect anomalous poses in real-time

### SVM Anomaly Detection

Train an anomaly detection model from normal pose data:

```bash
# 1. Collect normal pose keypoints
python -m poseplay grab video.mp4 --plugins yolo_pose_plugin,keypoints_save_plugin --save

# 2. Train SVM model on collected keypoints
python poseplay/svm.py --csv keypoints.csv --model-path anomaly_model.pkl --grid-search

# 3. Use trained model for real-time anomaly detection
python -m poseplay grab rtsp://camera/stream --plugins yolo_pose_plugin,svm_anomaly_plugin --model-path anomaly_model.pkl
```

SVM Parameters:
- `--nu FLOAT`: Anomaly parameter (0 < nu <= 1, default: 0.1)
- `--kernel STR`: Kernel type (rbf, linear, poly, sigmoid, default: rbf)
- `--gamma STR`: Kernel coefficient (scale, auto, or float, default: scale)

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
└── lib/
    ├── yolo_pose_plugin.py      # YOLO pose detection plugin
    ├── keypoints_save_plugin.py # Keypoints CSV saving plugin
    └── svm_anomaly_plugin.py    # SVM anomaly detection plugin

docs/
└── plugin-development.md # Plugin development guide

tests/
├── test_image_grabber.py     # Unit tests for image grabber
├── test_plugins.py          # Unit tests for plugin system
└── test_svm_anomaly_plugin.py # Unit tests for SVM anomaly plugin
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