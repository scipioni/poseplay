"""Unit tests for image grabber components."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

from poseplay.image_grabber import (
    ImageGrabber,
    FileImageGrabber,
    DirectoryImageGrabber,
    VideoGrabber,
    RTSPGrabber,
    ImageGrabberFactory,
)


class TestImageGrabber(unittest.TestCase):
    """Test the base ImageGrabber class."""

    def test_abstract_methods(self):
        """Test that ImageGrabber cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ImageGrabber()


class TestFileImageGrabber(unittest.TestCase):
    """Test FileImageGrabber functionality."""

    def setUp(self):
        """Create a temporary image file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test_image.png")

        # Create a simple test image (100x100 RGB)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Red channel
        cv2.imwrite(self.image_path, test_image)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                pass  # Directory not empty, but that's okay for cleanup

    def test_valid_image_file(self):
        """Test loading a valid image file."""
        grabber = FileImageGrabber(self.image_path)
        frame = grabber.get_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))
        self.assertEqual(frame.dtype, np.uint8)

        # Second call should return None
        self.assertIsNone(grabber.get_frame())

        grabber.close()

    def test_invalid_file_path(self):
        """Test error handling for non-existent file."""
        with self.assertRaises(ValueError):
            FileImageGrabber("/non/existent/file.png")

    def test_invalid_image_file(self):
        """Test error handling for invalid image content."""
        invalid_path = os.path.join(self.temp_dir, "invalid.txt")
        with open(invalid_path, "w") as f:
            f.write("not an image")

        with self.assertRaises(ValueError):
            grabber = FileImageGrabber(invalid_path)
            grabber.get_frame()


class TestDirectoryImageGrabber(unittest.TestCase):
    """Test DirectoryImageGrabber functionality."""

    def setUp(self):
        """Create a temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test images
        for i in range(3):
            image_path = os.path.join(self.temp_dir, f"test_image_{i}.png")
            test_image = np.full((50, 50, 3), i * 85, dtype=np.uint8)
            cv2.imwrite(image_path, test_image)

        # Create a non-image file
        txt_path = os.path.join(self.temp_dir, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("not an image")

    def tearDown(self):
        """Clean up temporary files."""
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def test_directory_with_images(self):
        """Test processing a directory with multiple images."""
        grabber = DirectoryImageGrabber(self.temp_dir)

        frames = []
        while True:
            frame = grabber.get_frame()
            if frame is None:
                break
            frames.append(frame)

        self.assertEqual(len(frames), 3)
        for frame in frames:
            self.assertEqual(frame.shape, (50, 50, 3))
            self.assertEqual(frame.dtype, np.uint8)

        grabber.close()

    def test_empty_directory(self):
        """Test handling of empty directory."""
        empty_dir = tempfile.mkdtemp()
        grabber = DirectoryImageGrabber(empty_dir)

        self.assertIsNone(grabber.get_frame())
        grabber.close()
        os.rmdir(empty_dir)

    def test_invalid_directory(self):
        """Test error handling for non-existent directory."""
        with self.assertRaises(ValueError):
            DirectoryImageGrabber("/non/existent/directory")


class TestVideoGrabber(unittest.TestCase):
    """Test VideoGrabber functionality."""

    def setUp(self):
        """Create a temporary video file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_video.mp4")

        # Create a simple test video (10 frames, 10x10)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.video_path, fourcc, 10.0, (10, 10))

        for i in range(10):
            frame = np.full((10, 10, 3), i * 25, dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                pass  # Directory not empty, but that's okay for cleanup

    def test_valid_video_file(self):
        """Test processing a valid video file."""
        grabber = VideoGrabber(self.video_path)

        frames = []
        while True:
            frame = grabber.get_frame()
            if frame is None:
                break
            frames.append(frame)

        self.assertEqual(len(frames), 10)
        for frame in frames:
            self.assertEqual(frame.shape, (10, 10, 3))
            self.assertEqual(frame.dtype, np.uint8)

        grabber.close()

    def test_invalid_video_path(self):
        """Test error handling for non-existent video file."""
        with self.assertRaises(ValueError):
            VideoGrabber("/non/existent/video.mp4")

    def test_corrupted_video_file(self):
        """Test error handling for corrupted video file."""
        corrupted_path = os.path.join(self.temp_dir, "corrupted.mp4")
        with open(corrupted_path, "w") as f:
            f.write("not a video")

        with self.assertRaises(ValueError):
            VideoGrabber(corrupted_path)


class TestRTSPGrabber(unittest.TestCase):
    """Test RTSPGrabber functionality."""

    @patch("cv2.VideoCapture")
    def test_valid_rtsp_url(self, mock_cv2):
        """Test RTSP grabber with mocked connection."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap

        grabber = RTSPGrabber("rtsp://example.com/stream")
        frame = grabber.get_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))
        grabber.close()

    @patch("cv2.VideoCapture")
    def test_connection_failure(self, mock_cv2):
        """Test handling of connection failures."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        with self.assertRaises(ValueError):
            RTSPGrabber("rtsp://example.com/stream")


class TestImageGrabberFactory(unittest.TestCase):
    """Test ImageGrabberFactory functionality."""

    def setUp(self):
        """Create temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test.png")
        self.video_path = os.path.join(self.temp_dir, "test.mp4")

        # Create test files
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(self.image_path, test_image)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.video_path, fourcc, 1.0, (10, 10))
        out.write(test_image)
        out.release()

    def tearDown(self):
        """Clean up temporary files."""
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def test_create_file_grabber(self):
        """Test factory creates FileImageGrabber for image files."""
        grabber = ImageGrabberFactory.create(self.image_path)
        self.assertIsInstance(grabber, FileImageGrabber)

    def test_create_directory_grabber(self):
        """Test factory creates DirectoryImageGrabber for directories."""
        grabber = ImageGrabberFactory.create(self.temp_dir)
        self.assertIsInstance(grabber, DirectoryImageGrabber)

    def test_create_video_grabber(self):
        """Test factory creates VideoGrabber for video files."""
        grabber = ImageGrabberFactory.create(self.video_path)
        self.assertIsInstance(grabber, VideoGrabber)

    @patch("cv2.VideoCapture")
    def test_create_rtsp_grabber(self, mock_cv2):
        """Test factory creates RTSPGrabber for RTSP URLs."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.return_value = mock_cap

        grabber = ImageGrabberFactory.create("rtsp://example.com/stream")
        self.assertIsInstance(grabber, RTSPGrabber)

    def test_unsupported_source(self):
        """Test factory raises error for unsupported sources."""
        with self.assertRaises(ValueError):
            ImageGrabberFactory.create("unsupported.txt")


if __name__ == "__main__":
    unittest.main()
