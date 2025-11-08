# PosePlay

A flexible pose detection application that supports various image input sources including files, directories, videos, and RTSP streams.

## Features

- **Multiple Input Sources**: Support for single images, image directories, video files, and RTSP streams
- **Unified Interface**: Consistent API across all input types
- **Real-time Processing**: Frame-by-frame processing with configurable frame rates
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

### Keyboard Controls

During playback (except RTSP streams):
- `q` or `ESC`: Quit the application
- `p` or `SPACE`: Pause/resume playback
- `r`: Reset the grabber (restart from beginning)

## API Usage

```python
from poseplay.image_grabber import ImageGrabberFactory

# Create a grabber for any supported source
grabber = ImageGrabberFactory.create("path/to/source")

# Process frames
while True:
    frame = grabber.get_frame()
    if frame is None:
        break
    # Process your frame here
    # frame is a numpy array with shape (height, width, 3), dtype uint8

grabber.close()
```

## Supported Formats

### Image Files
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Video Files
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

### RTSP Streams
- Any valid RTSP URL (rtsp://...)

## Development

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
└── ...

tests/
└── test_image_grabber.py # Unit tests
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