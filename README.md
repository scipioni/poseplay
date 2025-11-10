# PosePlay

A flexible pose detection application that supports various image input sources including files, directories, videos, and RTSP streams.


```
pose grab data/rooms/termica.mkv --save # --> data/rooms/termica.csv
svmtrain --csv data/rooms/termica.csv --grid-search # --> data/rooms/termica.pkl
```



## Features

- **Multiple Input Sources**: Support for single images, image directories, video files, and RTSP streams
- **Unified Interface**: Consistent API across all input types
- **Real-time Processing**: Frame-by-frame processing with configurable frame rates
- **Plugin System**: Extensible plugin architecture for custom image processing
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
└── ...


docs/
└── plugin-development.md # Plugin development guide

tests/
├── test_image_grabber.py # Unit tests for image grabber
└── test_plugins.py      # Unit tests for plugin system
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