# Change: Add Plugin Library System

## Why
The current PosePlay application has a loop grabbing image sources. To enable extensibility and allow third-party developers to add custom image processing without modifying core code, we need a plugin system that can dynamically load and manage image processor plugins.

## What Changes
- Add a new plugin system capability that allows loading image processor plugins from a plugins directory
- Implement plugin discovery, loading, and automatic registration mechanisms
- Add configuration support for enabling/disabling plugins
- Create plugin metadata and interface definitions
- Add dummy plugin implementation
- Plugins will have a processing incoming frame and annotating frame called from main loop

## Impact
- Affected specs: plugin-system (new), main (modified)
- Affected code: main.py, poseplay/config.py, new poseplay/plugins/ module
- No breaking changes to existing API