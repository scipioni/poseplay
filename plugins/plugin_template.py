"""Plugin Template

Copy this file and modify it to create your own plugin.
Rename the file to your_plugin_name.py and update the class name accordingly.
"""

from poseplay.plugins import Plugin, PluginMetadata
import numpy as np


class YourPluginName(Plugin):
    """Your plugin description here."""

    def __init__(self):
        # Initialize your plugin state here
        pass

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="your_plugin_name",  # Change this to your plugin name
            version="1.0.0",
            description="Brief description of what your plugin does",
            capabilities=["image_processor"]
        )

    def initialize(self) -> None:
        """Initialize the plugin. Called once when the plugin is loaded."""
        print(f"Initializing {self.metadata.name}")
        # Add any initialization logic here
        # - Load models
        # - Initialize connections
        # - Set up data structures

    def cleanup(self) -> None:
        """Clean up plugin resources. Called when the application shuts down."""
        print(f"Cleaning up {self.metadata.name}")
        # Add cleanup logic here
        # - Close connections
        # - Save state
        # - Free resources

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the frame and return the modified frame.

        Args:
            frame: Input frame as numpy array with shape (height, width, 3) and dtype uint8

        Returns:
            Processed frame as numpy array with same shape and dtype
        """
        # Your frame processing logic here
        # Examples:
        # - Add annotations/text overlays
        # - Apply image filters
        # - Detect objects
        # - Perform analysis

        # For now, just return the frame unchanged
        return frame