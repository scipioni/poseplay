# image-grabber Specification

## Purpose
TBD - created by archiving change add-image-grabber. Update Purpose after archive.
## Requirements
### Requirement: Image Grabber Base Interface
The system SHALL provide a base ImageGrabber interface that defines methods for extracting frames from various sources.

#### Scenario: Frame Extraction
- **WHEN** an ImageGrabber is initialized with a valid source
- **THEN** it SHALL provide a method to get the next frame as a numpy array
- **AND** it SHALL return None when no more frames are available
- **ACCEPTANCE CRITERIA**: Frame is returned as BGR numpy array with shape (height, width, 3), dtype uint8, or None

### Requirement: File Image Grabber
The system SHALL support grabbing images from individual image files.

#### Scenario: Single Image File
- **WHEN** a FileImageGrabber is created with a path to a valid image file (e.g., .jpg, .png)
- **THEN** get_frame() SHALL return the image as a numpy array on first call
- **AND** subsequent calls SHALL return None
- **ACCEPTANCE CRITERIA**: Supports common image formats, raises ValueError for invalid files

### Requirement: Directory Image Grabber
The system SHALL support grabbing images from a directory containing multiple image files.

#### Scenario: Directory of Images
- **WHEN** a DirectoryImageGrabber is created with a path to a directory containing image files
- **THEN** get_frame() SHALL return images one by one in alphabetical order
- **AND** it SHALL skip non-image files (e.g., .txt, .json)
- **AND** return None when all valid images are processed
- **ACCEPTANCE CRITERIA**: Processes .jpg, .png, .bmp files; ignores others; handles empty directories gracefully

### Requirement: Video File Grabber
The system SHALL support grabbing frames from video files.

#### Scenario: Video File Processing
- **WHEN** a VideoGrabber is created with a path to a valid video file (e.g., .mp4, .avi)
- **THEN** get_frame() SHALL return frames sequentially at original frame rate
- **AND** it SHALL support optional frame rate control via parameter
- **AND** return None when video ends or file is invalid
- **ACCEPTANCE CRITERIA**: Uses OpenCV for video processing; handles corrupted files with error logging

### Requirement: RTSP Stream Grabber
The system SHALL support grabbing frames from RTSP video streams.

#### Scenario: RTSP Stream Processing
- **WHEN** an RTSPGrabber is created with a valid RTSP URL
- **THEN** get_frame() SHALL return live frames from the stream
- **AND** it SHALL handle connection errors by retrying up to 3 times with exponential backoff
- **AND** return None if connection fails permanently
- **ACCEPTANCE CRITERIA**: Supports rtsp:// URLs; logs connection status; frame rate matches stream

### Requirement: Image Grabber Factory
The system SHALL provide a factory to create appropriate grabbers based on input type.

#### Scenario: Automatic Grabber Selection
- **WHEN** ImageGrabberFactory.create() is called with a file path ending in image extension
- **THEN** it SHALL return a FileImageGrabber instance
- **WHEN** called with a directory path
- **THEN** it SHALL return a DirectoryImageGrabber instance
- **WHEN** called with a video file extension
- **THEN** it SHALL return a VideoGrabber instance
- **WHEN** called with an RTSP URL (starting with rtsp://)
- **THEN** it SHALL return an RTSPGrabber instance
- **ACCEPTANCE CRITERIA**: Factory raises ValueError for unsupported input types

