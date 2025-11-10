"""YOLO pose detection plugin for PosePlay."""

import cv2
import numpy as np
from ultralytics import YOLO

from poseplay.plugins import Plugin, PluginMetadata


class YOLOPosePlugin(Plugin):
    """Plugin that detects human poses using YOLO and annotates frames with keypoints and skeleton."""

    # COCO pose keypoints indices
    KEYPOINTS = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }

    # Skeleton connections (pairs of keypoint indices)
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]

    def __init__(self, confidence_threshold: float = 0.5, model_path: str = "yolo11m-pose.pt"):
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None
        self.current_poses = []
        self.initialize()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="yolo_pose_plugin",
            version="1.0.0",
            description="YOLO-based human pose detection plugin that annotates frames with keypoints and skeleton",
            capabilities=["pose_detector", "image_processor"]
        )

    def initialize(self) -> None:
        """Initialize the plugin by loading the YOLO pose model."""
        try:
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO pose model: {self.model_path}")
        except Exception as e:
            print(f"Failed to load YOLO pose model {self.model_path}: {e}")
            self.model = None

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.model:
            # YOLO models don't have explicit cleanup, but we can set to None
            self.model = None
            print("Cleaned up YOLO pose plugin")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the frame to detect poses and annotate with keypoints and skeleton.

        Args:
            frame: Input frame as numpy array with shape (height, width, 3) and dtype uint8

        Returns:
            Processed frame with pose annotations
        """
        if self.model is None:
            return frame

        try:
            # Run pose detection
            results = self.model(frame, conf=self.confidence_threshold)

            # Store current poses for potential external access
            self.current_poses = []

            # Process each detection
            for result in results:
                if result.keypoints is not None:
                    poses = self._extract_poses(result.boxes, result.keypoints)
                    self.current_poses.extend(poses)
                    self._draw_poses(frame, poses)

            return frame, poses
        except Exception as e:
            print(f"Error processing frame with YOLO pose: {e}")
            return frame, None

    def _extract_poses(self, boxes, keypoints):
        """Extract pose data with keypoints relative to bounding boxes.

        Args:
            boxes: Detection bounding boxes
            keypoints: Keypoint data

        Returns:
            List of pose dictionaries with relative keypoints
        """
        # Convert to numpy arrays if needed
        if hasattr(boxes, 'xyxy'):
            boxes_xyxy = boxes.xyxy.cpu().numpy()
        else:
            boxes_xyxy = boxes

        if hasattr(keypoints, 'xy'):
            kp_xy = keypoints.xy.cpu().numpy()
            kp_conf = keypoints.conf.cpu().numpy()
        else:
            kp_xy = keypoints
            kp_conf = None

        poses = []
        for i, (box, kp) in enumerate(zip(boxes_xyxy, kp_xy)):
            x1, y1, x2, y2 = box[:4]
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Convert keypoints to relative coordinates (0-1 within bounding box)
            relative_keypoints = {}
            for j, (x, y) in enumerate(kp):
                # Get confidence as scalar
                conf = float(kp_conf[i, j]) if kp_conf is not None else 1.0
                # if conf >= self.confidence_threshold:
                rel_x = (x - x1) / bbox_w if bbox_w > 0 else 0
                rel_y = (y - y1) / bbox_h if bbox_h > 0 else 0
                relative_keypoints[self.KEYPOINTS[j]] = {
                    'x': max(0, min(1, rel_x)),  # Clamp to 0-1
                    'y': max(0, min(1, rel_y)),
                    'confidence': conf
                }

            poses.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'keypoints': relative_keypoints,
                'confidence': float(box[4]) if len(box) > 4 else 1.0,
                'xy': [(float(r["x"]), float(r["y"])) for r in relative_keypoints.values()]
            })

        return poses

    def _draw_poses(self, frame: np.ndarray, poses: list) -> None:
        """Draw pose keypoints and skeleton on the frame.

        Args:
            frame: Frame to draw on
            poses: List of pose dictionaries with bbox and keypoints
        """
        for pose in poses:
            # Extract bbox
            x1, y1, x2, y2 = pose['bbox']

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Draw keypoints and skeleton using relative coordinates
            self._draw_keypoints_and_skeleton(frame, pose['keypoints'], x1, y1, x2, y2)

    def _draw_keypoints_and_skeleton(self, frame: np.ndarray, keypoints: dict,
                                   bbox_x1: float, bbox_y1: float,
                                   bbox_x2: float, bbox_y2: float) -> None:
        """Draw keypoints as circles and skeleton as lines.

        Args:
            frame: Frame to draw on
            keypoints: Dictionary of keypoint data with relative coordinates (0-1)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2: Bounding box coordinates
        """
        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1

        # Draw keypoints
        for kp_name, kp_data in keypoints.items():
            # Convert from relative coordinates (0-1 within bbox) to absolute frame coordinates
            abs_x = int(bbox_x1 + kp_data['x'] * bbox_w)
            abs_y = int(bbox_y1 + kp_data['y'] * bbox_h)

            # Determine color based on confidence
            if kp_data['confidence'] > 0.5:
                color = (0, 255, 0)  # Green for high confidence
            else:
                color = (0, 0, 255)  # Red for low confidence

            # Draw keypoint circle
            cv2.circle(frame, (abs_x, abs_y), 4, color, -1)

        # Draw skeleton connections
        for start_idx, end_idx in self.SKELETON:
            start_name = self.KEYPOINTS[start_idx]
            end_name = self.KEYPOINTS[end_idx]

            # Check if both keypoints exist in the detected keypoints
            if start_name not in keypoints or end_name not in keypoints:
                continue

            start_kp = keypoints[start_name]
            end_kp = keypoints[end_name]

            # Convert from relative coordinates to absolute frame coordinates
            start_x = int(bbox_x1 + start_kp['x'] * bbox_w)
            start_y = int(bbox_y1 + start_kp['y'] * bbox_h)
            end_x = int(bbox_x1 + end_kp['x'] * bbox_w)
            end_y = int(bbox_y1 + end_kp['y'] * bbox_h)

            start_point = (start_x, start_y)
            end_point = (end_x, end_y)

            cv2.line(frame, start_point, end_point, (255, 255, 0), 2)