# Change: Add Keypoints Save Plugin

## Why
The current YOLO pose detection plugin detects and visualizes keypoints but doesn't save them to file. Users need a way to persist pose data for analysis, training, or offline processing.

## What Changes
- Add a new plugin that saves keypoints (produced by another plugin already present) to csv file in relative coordinates format: x1,y1,x2,y2,...
- Add a method to plugin to set keypoints
- Configurable output file. If not specified, use same path as input file path with .csv extension

## Impact
- Affected specs: pose-detection (extends existing capability)
- Affected code: new plugin file in plugins/ directory
- No breaking changes to existing functionality