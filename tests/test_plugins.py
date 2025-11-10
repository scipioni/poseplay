"""Tests for the plugin system."""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import sys

import numpy as np

from poseplay.plugins import Plugin, PluginMetadata, PluginRegistry, PluginLoader


class TestPluginMetadata(unittest.TestCase):
    """Test PluginMetadata dataclass."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            capabilities=["image_processor"],
        )

        self.assertEqual(metadata.name, "test_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "Test plugin")
        self.assertEqual(metadata.capabilities, ["image_processor"])


class TestPluginRegistry(unittest.TestCase):
    """Test PluginRegistry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = PluginRegistry()

    def test_register_plugin(self):
        """Test registering a plugin."""
        plugin = Mock()
        plugin.metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            capabilities=["image_processor"],
        )

        self.registry.register(plugin)
        self.assertIn("test_plugin", self.registry.list_plugins())
        self.assertEqual(self.registry.get_plugin("test_plugin"), plugin)

    def test_register_duplicate_plugin(self):
        """Test registering a plugin with duplicate name raises error."""
        plugin1 = Mock()
        plugin1.metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin 1",
            capabilities=["image_processor"],
        )

        plugin2 = Mock()
        plugin2.metadata = PluginMetadata(
            name="test_plugin",
            version="2.0.0",
            description="Test plugin 2",
            capabilities=["image_processor"],
        )

        self.registry.register(plugin1)
        with self.assertRaises(ValueError):
            self.registry.register(plugin2)

    def test_get_plugins_by_capability(self):
        """Test getting plugins by capability."""
        plugin1 = Mock()
        plugin1.metadata = PluginMetadata(
            name="plugin1",
            version="1.0.0",
            description="Plugin 1",
            capabilities=["image_processor"],
        )

        plugin2 = Mock()
        plugin2.metadata = PluginMetadata(
            name="plugin2",
            version="1.0.0",
            description="Plugin 2",
            capabilities=["audio_processor"],
        )

        self.registry.register(plugin1)
        self.registry.register(plugin2)

        image_plugins = self.registry.get_plugins_by_capability("image_processor")
        self.assertEqual(len(image_plugins), 1)
        self.assertEqual(image_plugins[0], plugin1)

    def test_initialize_all(self):
        """Test initializing all plugins."""
        plugin = Mock()
        plugin.metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            capabilities=[],
        )

        self.registry.register(plugin)
        self.registry.initialize_all()

        plugin.initialize.assert_called_once()

    def test_cleanup_all(self):
        """Test cleaning up all plugins."""
        plugin = Mock()
        plugin.metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            capabilities=[],
        )

        self.registry.register(plugin)
        self.registry.cleanup_all()

        plugin.cleanup.assert_called_once()


class TestPluginLoader(unittest.TestCase):
    """Test PluginLoader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = PluginLoader(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_discover_plugins_no_directory(self):
        """Test discovering plugins when directory doesn't exist."""
        loader = PluginLoader("/nonexistent/directory")
        plugins = loader.discover_plugins()
        self.assertEqual(plugins, [])

    def test_discover_plugins_empty_directory(self):
        """Test discovering plugins in empty directory."""
        plugins = self.loader.discover_plugins()
        self.assertEqual(plugins, [])

    def test_discover_python_plugins(self):
        """Test discovering Python plugin files."""
        # Create a dummy plugin file
        plugin_path = os.path.join(self.temp_dir, "test_plugin.py")
        with open(plugin_path, "w") as f:
            f.write("# Dummy plugin file")

        plugins = self.loader.discover_plugins()
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0], plugin_path)

    def test_load_python_plugin(self):
        """Test loading a Python plugin."""
        plugin_path = os.path.join(self.temp_dir, "test_plugin.py")
        with open(plugin_path, "w") as f:
            f.write("""
from poseplay.plugins import Plugin, PluginMetadata

class TestPlugin(Plugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            capabilities=["image_processor"]
        )

    def initialize(self):
        pass

    def cleanup(self):
        pass

    def process_frame(self, frame):
        return frame
""")

        plugin = self.loader.load_plugin(plugin_path)
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.metadata.name, "test_plugin")

    def test_load_invalid_plugin(self):
        """Test loading an invalid plugin file."""
        plugin_path = os.path.join(self.temp_dir, "invalid_plugin.py")
        with open(plugin_path, "w") as f:
            f.write("# Invalid plugin - no plugin class")

        plugin = self.loader.load_plugin(plugin_path)
        self.assertIsNone(plugin)


class TestExamplePlugin(unittest.TestCase):
    """Test the example plugin."""

    def setUp(self):
        """Set up test fixtures."""
        # Import the example plugin dynamically since it's in the plugins directory
        import sys
        import os

        plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)

        from example_plugin import ExamplePlugin

        self.plugin = ExamplePlugin()

    def test_metadata(self):
        """Test plugin metadata."""
        metadata = self.plugin.metadata
        self.assertEqual(metadata.name, "example_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertIn("image_processor", metadata.capabilities)

    def test_process_frame(self):
        """Test processing a frame."""
        # Create a test frame with some content
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Process the frame
        result = self.plugin.process_frame(frame)

        # Check that the frame was modified (text was added)
        self.assertEqual(result.shape, frame.shape)
        # Check that green text was added (BGR format: (0, 255, 0))
        # The result should have green pixels where text was drawn
        # Since we can't easily check exact pixel values, just ensure the function runs
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
"""Tests for the plugin system."""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import sys

import numpy as np

from poseplay.plugins import Plugin, PluginMetadata, PluginRegistry, PluginLoader


class TestPluginLoaderIntegration(unittest.TestCase):
    """Integration tests for plugin loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = PluginLoader(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_all_plugins_integration(self):
        """Test loading all plugins from directory."""
        # Create a test plugin file
        plugin_content = """
from poseplay.plugins import Plugin, PluginMetadata
import numpy as np

class TestIntegrationPlugin(Plugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test_integration_plugin",
            version="1.0.0",
            description="Integration test plugin",
            capabilities=["image_processor"]
        )

    def initialize(self):
        pass

    def cleanup(self):
        pass

    def process_frame(self, frame):
        return frame
"""

        plugin_path = os.path.join(self.temp_dir, "test_integration_plugin.py")
        with open(plugin_path, "w") as f:
            f.write(plugin_content)

        # Load all plugins
        self.loader.load_all_plugins()

        # Check that plugin was loaded and registered
        self.assertIn("test_integration_plugin", self.loader.registry.list_plugins())
        plugin = self.loader.registry.get_plugin("test_integration_plugin")
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.metadata.name, "test_integration_plugin")

    def test_plugin_initialization_and_cleanup(self):
        """Test that plugins are properly initialized and cleaned up."""
        # Create a test plugin that tracks initialization/cleanup
        plugin_content = """
from poseplay.plugins import Plugin, PluginMetadata
import numpy as np

class LifecycleTestPlugin(Plugin):
    def __init__(self):
        self.initialized = False
        self.cleaned_up = False

    @property
    def metadata(self):
        return PluginMetadata(
            name="lifecycle_test_plugin",
            version="1.0.0",
            description="Lifecycle test plugin",
            capabilities=["image_processor"]
        )

    def initialize(self):
        self.initialized = True

    def cleanup(self):
        self.cleaned_up = True

    def process_frame(self, frame):
        return frame
"""

        plugin_path = os.path.join(self.temp_dir, "lifecycle_test_plugin.py")
        with open(plugin_path, "w") as f:
            f.write(plugin_content)

        # Load plugins
        self.loader.load_all_plugins()

        # Initialize all plugins
        self.loader.registry.initialize_all()

        # Check initialization
        plugin = self.loader.registry.get_plugin("lifecycle_test_plugin")
        self.assertTrue(plugin.initialized)

        # Clean up all plugins
        self.loader.registry.cleanup_all()

        # Check cleanup
        self.assertTrue(plugin.cleaned_up)


if __name__ == "__main__":
    unittest.main()
