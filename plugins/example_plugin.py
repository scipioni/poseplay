"""Example plugin demonstrating basic image processing capabilities."""

import cv2
import numpy as np

from poseplay.plugins import Plugin, PluginMetadata


class ExamplePlugin(Plugin):
    """Example plugin that adds a simple overlay to frames."""

    def __init__(self):
        self.overlay_text = "Plugin Active"

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin that adds overlay text to frames",
            capabilities=["image_processor"]
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        print(f"Initializing {self.metadata.name}")

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"Cleaning up {self.metadata.name}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add overlay text to the frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Frame with overlay text added
        """
        # Add overlay text
        cv2.putText(
            frame,
            self.overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        return frame