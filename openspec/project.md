# Project Context

## Purpose
PosePlay is a Python-based application for real-time pose detection and camera-aware fall detection. The project aims to provide an efficient, GPU-accelerated solution for monitoring human poses from video feeds, with a focus on detecting falls to enhance safety in environments like homes or care facilities. It leverages advanced computer vision models to process camera input and alert on potential fall events.

## Tech Stack
- **Language**: Python 3.7+
- **Core Libraries**:
  - PyTorch (for deep learning and GPU acceleration)
  - OpenCV (for computer vision and image processing)
  - Ultralytics YOLO (for pose detection)
  - MediaPipe (for pose estimation and tracking)
  - NumPy (for numerical computations)
  - scikit-learn (for machine learning utilities)
- **Configuration**: PyYAML (for configuration management)
- **Build System**: setuptools (for packaging)
- **Containerization**: Docker with NVIDIA GPU support
- **Development Tools**: pytest, black, flake8, mypy (for testing and code quality)

## Project Conventions

### Code Style
- **Formatter**: Black (line length 88 characters)
- **Linter**: flake8 (with default rules)
- **Type Checker**: mypy (strict mode preferred)
- **Naming Conventions**:
  - Functions and variables: snake_case
  - Classes: PascalCase
  - Constants: UPPER_SNAKE_CASE
  - Modules: snake_case
- **Import Organization**: Standard library, third-party, local imports (separated by blank lines)
- **Docstrings**: Use Google-style docstrings for functions and classes

### Architecture Patterns
- **Modular Design**: Single responsibility principle with separate modules for pose detection, fall detection, and camera handling
- **Pipeline Architecture**: Data flows through detection → processing → alerting stages
- **GPU Acceleration**: Leverage PyTorch and CUDA for model inference
- **Configuration-Driven**: Use YAML for runtime configuration without code changes
- **Containerized Deployment**: Docker with NVIDIA GPU support for consistent environments
- **CLI Interface**: Command-line entry point for easy integration and scripting

### Testing Strategy
- **Framework**: pytest for unit and integration tests
- **Coverage**: pytest-cov for coverage reporting (target: >80%)
- **Test Organization**: Tests in separate directory mirroring source structure
- **Types of Tests**:
  - Unit tests for individual functions and classes
  - Integration tests for pipeline components
  - Mock external dependencies (camera, models) for CI/CD
- **Containerization**: run python commands with "docker compose run --rm pose <command>"

### Git Workflow
- **Branching Strategy**: Git Flow with main/master as production, develop for integration, feature branches for new work
- **Commit Conventions**: Conventional commits (feat:, fix:, docs:, style:, refactor:, test:, chore:)
- **Pull Requests**: Required for all changes, with code review
- **Release Process**: Tag releases from main branch, use semantic versioning

## Domain Context
- **Computer Vision**: Understanding of pose estimation, keypoint detection, and human pose analysis
- **Fall Detection**: Knowledge of fall patterns, camera perspectives, and safety monitoring systems
- **Real-time Processing**: Requirements for low-latency video processing and GPU acceleration
- **Camera Systems**: Familiarity with camera calibration, field of view, and perspective correction
- **Machine Learning Models**: Experience with YOLO architecture and MediaPipe pose estimation

## Important Constraints
- **GPU Requirements**: NVIDIA GPU with CUDA support for real-time pose detection
- **Camera Access**: Requires camera device access (/dev/video0) for live video input
- **Performance**: Must process video at real-time rates (30+ FPS) for effective fall detection
- **Privacy**: Handle video data securely, avoid unnecessary storage of sensitive footage
- **Resource Usage**: Balance model accuracy with computational efficiency for edge deployment
- **Containerization**: run python commands with "docker compose run --rm pose <command>"

## External Dependencies
- **YOLO Model**: Pre-trained YOLO11m pose model (yolo11m-pose.pt) for pose detection
- **PyTorch Ecosystem**: CUDA runtime and cuDNN for GPU acceleration
- **Camera Hardware**: Video4Linux-compatible camera devices
- **Display Server**: X11 for GUI applications (when needed for debugging)
- **Container Runtime**: Docker with NVIDIA Container Toolkit for GPU support
