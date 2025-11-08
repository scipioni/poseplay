# Plugin Development Guide

This guide explains how to develop plugins for the PosePlay application.

## Overview

PosePlay supports a plugin system that allows third-party developers to extend the application's functionality. Plugins can process image frames in real-time as they are displayed.

## Plugin Architecture

### Base Plugin Class

All plugins must inherit from the `Plugin` base class defined in `poseplay.plugins`. Here's the interface:

```python
from poseplay.plugins import Plugin, PluginMetadata
import numpy as np

class MyPlugin(Plugin):
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="Description of what my plugin does",
            capabilities=["image_processor"]  # Currently only "image_processor" is supported
        )

    def initialize(self) -> None:
        """Initialize the plugin. Called once when the plugin is loaded."""
        pass

    def cleanup(self) -> None:
        """Clean up plugin resources. Called when the application shuts down."""
        pass

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the frame and return the modified frame.

        Args:
            frame: Input frame as numpy array with shape (height, width, 3) and dtype uint8

        Returns:
            Processed frame as numpy array with same shape and dtype
        """
        # Your processing logic here
        return frame
```

### Plugin Metadata

Each plugin must provide metadata including:
- `name`: Unique identifier for the plugin
- `version`: Version string (e.g., "1.0.0")
- `description`: Human-readable description
- `capabilities`: List of capabilities (currently only "image_processor")

### Processing Pipeline

Plugins are executed in the order they are discovered. Each plugin receives the output of the previous plugin as input. The processing happens in the main display loop:

1. Frame is grabbed from source
2. Frame is passed through each enabled plugin's `process_frame()` method
3. Processed frame is displayed

## Plugin Discovery

Plugins are automatically discovered from the `plugins/` directory. The system looks for:
- `.py` files containing plugin classes
- `.zip` archives containing plugin modules

## Example Plugin

See `plugins/example_plugin.py` for a complete example that adds overlay text to frames.

## Development Tips

1. **Frame Processing**: Always return a frame with the same shape and dtype as the input
2. **Performance**: Keep processing fast to maintain real-time performance
3. **Error Handling**: Handle exceptions gracefully to avoid breaking the main loop
4. **Resources**: Clean up any resources (files, connections, etc.) in the `cleanup()` method
5. **Thread Safety**: Be aware that `process_frame()` may be called from the main thread

## Testing

Test your plugin by placing it in the `plugins/` directory and running PosePlay. You can also write unit tests following the pattern in `tests/test_plugins.py`.

## Distribution

Plugins can be distributed as:
- Single `.py` files
- `.zip` archives containing the plugin module and dependencies

## Configuration

Plugin loading can be configured via command-line arguments:
- `--plugins-dir`: Directory to scan for plugins (default: "plugins")
- `--enabled-plugins`: List of specific plugins to enable (default: all discovered)

## Best Practices

1. Use descriptive plugin names and versions
2. Document your plugin's behavior and requirements
3. Handle edge cases (empty frames, corrupted data)
4. Test with different frame sizes and formats
5. Keep dependencies minimal to avoid conflicts