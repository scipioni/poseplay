# Change: Add General Purpose Image Grabber

## Why
The project currently lacks a flexible image grabbing component to handle various input sources like file lists, directories of images, videos, and RTSP streams. This is needed to support pose detection from diverse media sources beyond just live camera feeds.

## What Changes
- Add a new `image-grabber` capability to handle multiple input types
- Support file image lists, directory of images, video files, and RTSP streams
- Provide a unified interface for frame extraction across different sources
- Provide a configuration module that will allow users to configure input sources and parameters from cli
- Add command cli to grab image in a loop in main
- Add keyboard shortcuts to control image grabbing operations during runtime (pause, resume, and exit) but not for rtsp streams
- Documentation for the new image grabber module 

## Impact
- Affected specs: New `image-grabber` capability
- Affected code: New module in `poseplay/image_grabber.py`
- Affected code: New module in `poseplay/config.py`
- Affected documentation: `README.md`
- No breaking changes to existing functionality