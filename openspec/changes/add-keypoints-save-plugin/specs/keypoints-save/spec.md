## ADDED Requirements
### Requirement: Keypoints Save Plugin
The system SHALL provide a plugin that saves detected pose keypoints to CSV files for offline analysis and processing.

#### Scenario: Save Keypoints to File
- **WHEN** the keypoints_save_plugin receives keypoints via set_keypoints method
- **THEN** it SHALL save keypoint data to a CSV file
- **ACCEPTANCE CRITERIA**: CSV file contains one row per pose without header row; format is x1,y1,x2,y2,... where coordinates are relative (0-1 normalized to image dimensions); output file is configurable, if not specified uses same path as input file with .csv extension

#### Scenario: Plugin Metadata
- **WHEN** the plugin is loaded
- **THEN** it SHALL provide metadata with name "keypoints_save_plugin", version, description, and "keypoints_saver" capability
- **ACCEPTANCE CRITERIA**: Metadata accessible via plugin.metadata property; capabilities list includes "keypoints_saver"; plugin inherits from base Plugin class

#### Scenario: Set Keypoints Method
- **WHEN** another plugin provides keypoints
- **THEN** the keypoints_save_plugin SHALL have a set_keypoints method to receive the keypoints data
- **ACCEPTANCE CRITERIA**: set_keypoints method accepts keypoints as list of poses, each pose as list of (x,y) coordinates; coordinates are relative 0-1; method stores keypoints for saving to file

#### Scenario: Configurable Output
- **WHEN** the plugin initializes
- **THEN** it SHALL use configurable output file path
- **AND** if not specified, derive from input file path
- **ACCEPTANCE CRITERIA**: Configuration allows setting output_file_path; if not set, uses input_file_path with extension replaced by .csv; handles file writing errors gracefully by logging warning