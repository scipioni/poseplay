"""Configuration module for poseplay CLI."""

import argparse
from typing import Any, Dict, List
import logging


class Config:
    """Configuration class for poseplay application."""

    def __init__(self):
        self.source: str = ""
        self.fps: float = 30.0
        self.max_retries: int = 3
        self.loop: bool = False
        self.plugins_dir: str = "plugins"
        self.save: bool = False
        self.enabled_plugins: List[str] = []
        self.plugin_config: Dict[str, Any] = {}
        self.svm: str = ""
        self.autoencoder: str = ""


    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create Config from parsed arguments."""
        config = cls()
        config.source = args.source
        config.fps = args.fps
        config.max_retries = args.max_retries
        config.loop = args.loop
        config.plugins_dir = getattr(args, 'plugins_dir', 'plugins')
        config.enabled_plugins = getattr(args, 'enabled_plugins', [])
        config.save = args.save
        config.svm = args.svm
        config.autoencoder = args.autoencoder
        return config

    def to_grabber_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs for ImageGrabberFactory.create()."""
        kwargs = {}
        if self.fps != 30.0:
            kwargs["fps"] = self.fps
        if self.max_retries != 3:
            kwargs["max_retries"] = self.max_retries
        return kwargs


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for poseplay CLI."""
    parser = argparse.ArgumentParser(
        description="Pose detection from various image sources", prog="poseplay"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # grab command
    grab_parser = subparsers.add_parser("grab", help="Grab images from source")
    grab_parser.add_argument(
        "source", help="Source: image file, directory, video file, or RTSP URL"
    )
    grab_parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate for video processing (default: 30.0)",
    )
    grab_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for RTSP connection (default: 3)",
    )
    grab_parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop continuously (restart when source ends)",
    )
    grab_parser.add_argument(
        "--plugins-dir",
        default="plugins",
        help="Directory containing plugins (default: plugins)",
    )
    grab_parser.add_argument(
        "--enabled-plugins",
        nargs="*",
        default=[],
        help="List of enabled plugin names (default: all discovered plugins)",
    )
    grab_parser.add_argument(
        "--save",
        action="store_true",
        help="Loop continuously (restart when source ends)",
    )
    grab_parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode for logging (default: INFO)",
    )
    grab_parser.add_argument(
        "--svm",
        default="",
        help="svm model file for pose detection",
    )
    grab_parser.add_argument(
        "--autoencoder",
        default="",
        help="autoencoder model file for pose anomaly detection",
    )
    return parser


def parse_args() -> Config:
    """Parse command line arguments and return Config."""
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        level="DEBUG" if args.debug else "INFO",
    )

    if args.command == "grab":
        return Config.from_args(args)
    else:
        parser.print_help()
        raise SystemExit(1)
