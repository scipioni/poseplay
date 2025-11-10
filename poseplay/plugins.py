"""Plugin system for poseplay application."""

import dataclasses
import importlib.util
import logging
import os
import sys
import zipfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)
"""Plugin system for poseplay application."""

import dataclasses
from abc import ABC, abstractmethod
from typing import List
import numpy as np


@dataclasses.dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    description: str
    capabilities: List[str]  # e.g., ["image_processor"]


class Plugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
