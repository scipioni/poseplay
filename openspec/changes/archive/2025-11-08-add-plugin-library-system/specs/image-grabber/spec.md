## MODIFIED Requirements

### Requirement: Image Grabber Factory
The system SHALL provide a factory to create appropriate grabbers based on input type, including plugin-based grabbers.

#### Scenario: Automatic Grabber Selection
- **WHEN** ImageGrabberFactory.create() is called with a file path ending in image extension
- **THEN** it SHALL return a FileImageGrabber instance
- **WHEN** called with a directory path
- **THEN** it SHALL return a DirectoryImageGrabber instance
- **WHEN** called with a video file extension
- **THEN** it SHALL return a VideoGrabber instance
- **WHEN** called with an RTSP URL (starting with rtsp://)
- **THEN** it SHALL return an RTSPGrabber instance
- **WHEN** called with a plugin-supported source type
- **THEN** it SHALL delegate to the appropriate plugin to create the grabber
- **ACCEPTANCE CRITERIA**: Factory raises ValueError for unsupported input types; plugin grabbers integrate seamlessly

### Requirement: Image Grabber Base Interface
The system SHALL provide a base ImageGrabber interface that defines methods for extracting frames from various sources, including plugin sources.

#### Scenario: Frame Extraction
- **WHEN** an ImageGrabber is initialized with a valid source
- **THEN** it SHALL provide a method to get the next frame as a numpy array
- **AND** it SHALL return None when no more frames are available
- **ACCEPTANCE CRITERIA**: Frame is returned as BGR numpy array with shape (height, width, 3), dtype uint8, or None; plugin grabbers implement this interface