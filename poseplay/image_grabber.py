"""General purpose image grabber for various input sources."""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Union
import cv2
import numpy as np


class ImageGrabber(ABC):
    """Abstract base class for image grabbers."""

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the source.

        Returns:
            numpy array with shape (height, width, 3) and dtype uint8, or None if no more frames
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass


class FileImageGrabber(ImageGrabber):
    """Grabber for single image files."""

    def __init__(self, file_path: str):
        """Initialize with image file path.

        Args:
            file_path: Path to image file (.jpg, .png, etc.)

        Raises:
            ValueError: If file doesn't exist or is not a valid image
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File does not exist: {file_path}")

        self.file_path = file_path
        self._frame_returned = False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the image frame.

        Returns:
            numpy array with shape (height, width, 3) and dtype uint8, or None if already returned
        """
        if self._frame_returned:
            return None

        image = cv2.imread(self.file_path)
        if image is None:
            raise ValueError(f"Could not read image file: {self.file_path}")

        self._frame_returned = True
        return image

    def close(self) -> None:
        """Clean up resources (no-op for file grabber)."""
        pass


class DirectoryImageGrabber(ImageGrabber):
    """Grabber for directories containing multiple images."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, directory_path: str):
        """Initialize with directory path.

        Args:
            directory_path: Path to directory containing images

        Raises:
            ValueError: If directory doesn't exist
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")

        self.directory_path = directory_path
        self._image_files = self._get_image_files()
        self._current_index = 0

    def _get_image_files(self) -> list[str]:
        """Get sorted list of image files in directory."""
        files = []
        for filename in os.listdir(self.directory_path):
            if os.path.isfile(os.path.join(self.directory_path, filename)):
                _, ext = os.path.splitext(filename.lower())
                if ext in self.SUPPORTED_EXTENSIONS:
                    files.append(filename)
        return sorted(files)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next image frame.

        Returns:
            numpy array with shape (height, width, 3) and dtype uint8, or None if no more images
        """
        if self._current_index >= len(self._image_files):
            return None

        file_path = os.path.join(
            self.directory_path, self._image_files[self._current_index]
        )
        image = cv2.imread(file_path)
        if image is None:
            # Skip corrupted files
            self._current_index += 1
            return self.get_frame()

        self._current_index += 1
        return image

    def close(self) -> None:
        """Clean up resources (no-op for directory grabber)."""
        pass


class VideoGrabber(ImageGrabber):
    """Grabber for video files."""

    def __init__(self, video_path: str, fps: Optional[float] = None):
        """Initialize with video file path.

        Args:
            video_path: Path to video file (.mp4, .avi, etc.)
            fps: Optional frame rate control (if None, uses video's native fps)

        Raises:
            ValueError: If video file doesn't exist or can't be opened
        """
        if not os.path.isfile(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = fps
        if self.fps is None:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_delay = 1.0 / self.fps if self.fps > 0 else 0
        self._last_frame_time: float = 0.0

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next video frame.

        Returns:
            numpy array with shape (height, width, 3) and dtype uint8, or None if video ended
        """
        if self.fps > 0:
            current_time = time.time()
            if current_time - self._last_frame_time < self._frame_delay:
                time.sleep(self._frame_delay - (current_time - self._last_frame_time))

        ret, frame = self.cap.read()
        if not ret:
            return None

        self._last_frame_time = time.time()
        return frame

    def close(self) -> None:
        """Release video capture."""
        if self.cap.isOpened():
            self.cap.release()


class RTSPGrabber(ImageGrabber):
    """Grabber for RTSP video streams."""

    def __init__(self, rtsp_url: str, max_retries: int = 3):
        """Initialize with RTSP URL.

        Args:
            rtsp_url: RTSP stream URL (rtsp://...)
            max_retries: Maximum reconnection attempts
        """
        self.rtsp_url = rtsp_url
        self.max_retries = max_retries
        self.cap: Optional[cv2.VideoCapture] = None
        self._connect()

    def _connect(self) -> None:
        """Establish RTSP connection with retries."""
        for attempt in range(self.max_retries):
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                return
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff

        raise ValueError(f"Could not connect to RTSP stream: {self.rtsp_url}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from RTSP stream.

        Returns:
            numpy array with shape (height, width, 3) and dtype uint8, or None if connection failed
        """
        if self.cap is None or not self.cap.isOpened():
            try:
                self._connect()
            except ValueError:
                return None

        ret, frame = self.cap.read()
        if not ret:
            # Try to reconnect once
            try:
                self._connect()
                ret, frame = self.cap.read()
                if not ret:
                    return None
            except ValueError:
                return None

        return frame

    def close(self) -> None:
        """Release RTSP capture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()


class ImageGrabberFactory:
    """Factory for creating appropriate image grabbers."""

    @staticmethod
    def create(source: str, **kwargs) -> ImageGrabber:
        """Create an image grabber based on the source type.

        Args:
            source: File path, directory path, or RTSP URL
            **kwargs: Additional arguments for grabber initialization

        Returns:
            Appropriate ImageGrabber instance

        Raises:
            ValueError: If source type is not supported
        """
        if source.startswith("rtsp://"):
            return RTSPGrabber(source, **kwargs)

        if os.path.isfile(source):
            # Check if it's a video file
            _, ext = os.path.splitext(source.lower())
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
            if ext in video_extensions:
                return VideoGrabber(source, **kwargs)
            else:
                return FileImageGrabber(source, **kwargs)

        if os.path.isdir(source):
            return DirectoryImageGrabber(source, **kwargs)

        raise ValueError(f"Unsupported source type: {source}")
