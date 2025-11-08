# Change: add-yolo-pose-plugin

## Why
The project currently has a plugin system but lacks a pose detection plugin using YOLO. Adding a YOLO pose detection plugin will enable pose estimation capabilities, which is core to the PosePlay application's purpose of image processing and pose analysis.

## What Changes
- Add a new plugin called `yolo_pose_plugin` that uses Ultralytics YOLO for pose detection with "pose" capabilities
- The plugin will detect human poses in frames and annotate them with keypoints and bounding boxes
- Keypoints coordinates must be relatives to bounding boxes and from 0 to 1 not to the frame origin
- Integrate with the existing plugin system architecture

## Impact
- Affected specs: plugin-system (adds new plugin capability)
- Affected code: new plugin file in plugins/ directory
- No Breaking changes to existing plugin interfaces
