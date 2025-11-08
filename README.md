# PosePlay

A flexible pose detection application that supports various image input sources including files, directories, videos, and RTSP streams.

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
- `--plugins-dir DIR`: Directory containing plugins (default: plugins)
- `--enabled-plugins LIST`: List of enabled plugin names (default: all discovered)

### Keyboard Controls

During playback (except RTSP streams):
- `q` or `ESC`: Quit the application
- `p` or `SPACE`: Pause/resume playback
- `r`: Reset the grabber (restart from beginning)

## Plugin System

PosePlay supports a plugin system for extending functionality with custom image processing. Plugins can add overlays, perform analysis, apply filters, and more.

### Using Plugins

Plugins are automatically loaded from the `plugins/` directory. Create your plugin by following the [Plugin Development Guide](docs/plugin-development.md).

### API Usage

```python
from poseplay.image_grabber import ImageGrabberFactory
from poseplay.plugins import PluginLoader

# Create a grabber for any supported source
grabber = ImageGrabberFactory.create("path/to/source")

# Initialize plugins
plugin_loader = PluginLoader("plugins")
plugin_loader.load_all_plugins()
plugin_loader.registry.initialize_all()

# Process frames with plugins
while True:
    frame = grabber.get_frame()
    if frame is None:
        break

    # Apply plugins
    processed_frame = frame
    for plugin in plugin_loader.registry.get_plugins_by_capability("image_processor"):
        processed_frame = plugin.process_frame(processed_frame)

    # Use processed_frame here

grabber.close()
plugin_loader.registry.cleanup_all()
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
├── plugins.py           # Plugin system implementation
└── ...

plugins/
├── example_plugin.py    # Example plugin implementation
└── plugin_template.py   # Template for creating new plugins

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