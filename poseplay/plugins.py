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

    # @abstractmethod
    # def process_frame(self, frame: np.ndarray) -> np.ndarray:
    #     """Process the frame and return the modified frame.

    #     Args:
    #         frame: Input frame as numpy array

    #     Returns:
    #         Processed frame as numpy array
    #     """
    #     pass
# class PluginRegistry:
#     """Registry for managing loaded plugins."""

#     def __init__(self):
#         self._plugins: Dict[str, Plugin] = {}

#     def register(self, plugin: Plugin) -> None:
#         """Register a plugin."""
#         name = plugin.metadata.name
#         if name in self._plugins:
#             raise ValueError(f"Plugin '{name}' already registered")
#         self._plugins[name] = plugin
#         logger.info(f"Registered plugin: {name}")

#     def get_plugin(self, name: str) -> Optional[Plugin]:
#         """Get a plugin by name."""
#         return self._plugins.get(name)

#     def list_plugins(self) -> List[str]:
#         """List all registered plugin names."""
#         return list(self._plugins.keys())

#     def get_plugins_by_capability(self, capability: str) -> List[Plugin]:
#         """Get plugins that have a specific capability."""
#         return [
#             plugin for plugin in self._plugins.values()
#             if capability in plugin.metadata.capabilities
#         ]

#     def initialize_all(self) -> None:
#         """Initialize all registered plugins."""
#         for plugin in self._plugins.values():
#             try:
#                 plugin.initialize()
#                 logger.info(f"Initialized plugin: {plugin.metadata.name}")
#             except Exception as e:
#                 logger.error(f"Failed to initialize plugin {plugin.metadata.name}: {e}")

#     def cleanup_all(self) -> None:
#         """Clean up all registered plugins."""
#         for plugin in self._plugins.values():
#             try:
#                 plugin.cleanup()
#                 logger.info(f"Cleaned up plugin: {plugin.metadata.name}")
#             except Exception as e:
#                 logger.error(f"Failed to cleanup plugin {plugin.metadata.name}: {e}")


# class PluginLoader:
#     """Handles loading and validation of plugins."""

#     def __init__(self, plugins_dir: str):
#         self.plugins_dir = plugins_dir
#         self.registry = PluginRegistry()

#     def discover_plugins(self) -> List[str]:
#         """Discover plugin files in the plugins directory."""
#         if not os.path.exists(self.plugins_dir):
#             logger.warning(f"Plugins directory does not exist: {self.plugins_dir}")
#             return []

#         plugin_files = []
#         for filename in os.listdir(self.plugins_dir):
#             filepath = os.path.join(self.plugins_dir, filename)
#             if filename.endswith('.py') and os.path.isfile(filepath):
#                 plugin_files.append(filepath)
#             elif filename.endswith('.zip') and os.path.isfile(filepath):
#                 plugin_files.append(filepath)

#         return plugin_files

#     def load_plugin(self, plugin_path: str) -> Optional[Plugin]:
#         """Load a plugin from file path."""
#         try:
#             if plugin_path.endswith('.py'):
#                 return self._load_python_plugin(plugin_path)
#             elif plugin_path.endswith('.zip'):
#                 return self._load_zip_plugin(plugin_path)
#             else:
#                 logger.warning(f"Unsupported plugin file type: {plugin_path}")
#                 return None
#         except Exception as e:
#             logger.error(f"Failed to load plugin {plugin_path}: {e}")
#             return None

#     def _load_python_plugin(self, plugin_path: str) -> Optional[Plugin]:
#         """Load a Python plugin file."""
#         module_name = os.path.splitext(os.path.basename(plugin_path))[0]

#         spec = importlib.util.spec_from_file_location(module_name, plugin_path)
#         if spec is None or spec.loader is None:
#             logger.error(f"Could not create module spec for {plugin_path}")
#             return None

#         module = importlib.util.module_from_spec(spec)
#         sys.modules[module_name] = module
#         spec.loader.exec_module(module)

#         # Find the plugin class (should be the only class inheriting from Plugin)
#         plugin_class = None
#         for attr_name in dir(module):
#             attr = getattr(module, attr_name)
#             if (isinstance(attr, type) and
#                 issubclass(attr, Plugin) and
#                 attr != Plugin):
#                 plugin_class = attr
#                 break

#         if plugin_class is None:
#             logger.error(f"No plugin class found in {plugin_path}")
#             return None

#         plugin_instance = plugin_class()
#         return plugin_instance

#     def _load_zip_plugin(self, plugin_path: str) -> Optional[Plugin]:
#         """Load a plugin from a zip archive."""
#         # For simplicity, extract to a temporary directory and load as Python
#         import tempfile
#         with tempfile.TemporaryDirectory() as temp_dir:
#             with zipfile.ZipFile(plugin_path, 'r') as zip_ref:
#                 zip_ref.extractall(temp_dir)

#             # Find the main plugin file (assume it's named after the zip)
#             base_name = os.path.splitext(os.path.basename(plugin_path))[0]
#             plugin_file = os.path.join(temp_dir, f"{base_name}.py")

#             if os.path.exists(plugin_file):
#                 return self._load_python_plugin(plugin_file)
#             else:
#                 logger.error(f"No main plugin file found in zip: {plugin_path}")
#                 return None

#     def load_all_plugins(self) -> None:
#         """Load all discovered plugins."""
#         plugin_files = self.discover_plugins()
#         for plugin_file in plugin_files:
#             plugin = self.load_plugin(plugin_file)
#             if plugin:
#                 try:
#                     self.registry.register(plugin)
#                 except ValueError as e:
#                     logger.warning(f"Could not register plugin: {e}")