"""Keypoints save plugin for PosePlay."""

import csv
import logging
import os
import time
from typing import List, Optional, TextIO, Tuple

from poseplay.plugins import Plugin, PluginMetadata

logger = logging.getLogger(__name__)


class KeypointsSavePlugin(Plugin):
    """Plugin that saves detected pose keypoints to a CSV file."""

    def __init__(
        self, input_file_path: str = "", output_file_path: Optional[str] = None
    ):
        self.input_file_path = input_file_path
        if output_file_path is None:
            # Derive output path from input path, replace extension with .csv
            base_name = os.path.splitext(input_file_path)[0]
            self.output_file_path = f"{base_name}.csv"
        else:
            self.output_file_path = output_file_path

        self.csv_writer: Optional[csv.writer] = None
        self.csv_file: Optional[TextIO] = None
        self.frame_number = 0

        self.initialize()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="keypoints_save_plugin",
            version="1.0.0",
            description="Plugin to save detected pose keypoints to CSV file",
            capabilities=["keypoints_saver"],
        )

    def initialize(self) -> None:
        """Initialize the plugin by opening the CSV file for writing."""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.csv_file = open(self.output_file_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            # No header row as per spec
            logger.info(
                f"Initialized keypoints save plugin, output: {self.output_file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize keypoints save plugin: {e}")
            self.csv_writer = None
            self.csv_file = None

    def set_output_file_path(self, output_file_path: str) -> None:
        """Set the output file path for saving keypoints."""
        self.output_file_path = output_file_path
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.csv_file = open(self.output_file_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            # No header row as per spec
            logger.info(
                f"Initialized keypoints save plugin, output: {self.output_file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize keypoints save plugin: {e}")
            self.csv_writer = None
            self.csv_file = None

    def cleanup(self) -> None:
        """Clean up plugin resources by closing the CSV file."""
        if self.csv_file:
            try:
                self.csv_file.close()
                logger.info(f"Save CSV {self.output_file_path}")
            except Exception as e:
                logger.error(f"Error closing CSV file: {e}")

    def process_frame(self, frame):
        """Process the frame (no-op for saver plugin)."""
        return frame

    def add(self, xy: List[List[Tuple[float, float]]]) -> None:
        """Set keypoints data and save to CSV.

        xy: List of poses, each pose as list of (x, y) coordinates (relative 0-1 to image dimensions)
        """
        if self.csv_writer is None:
            logger.warning("CSV writer not initialized, cannot save keypoints")
            return

        try:
            # timestamp = time.time()
            # for pose in xy:
            # Flatten the pose coordinates: x1,y1,x2,y2,...
            flattened_coords = [coord for pair in xy for coord in pair]
            # row = [self.frame_number, timestamp] + flattened_coords
            # print(flattened_coords)
            self.csv_writer.writerow(flattened_coords)

            self.frame_number += 1
        except Exception as e:
            logger.warning(f"Failed to save keypoints to CSV: {e}")
