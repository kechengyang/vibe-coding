# Changelog

All notable changes to the Employee Health Monitoring System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Video capture module with webcam support
- Posture detection using MediaPipe
- Drinking detection using MediaPipe
- SQLite database for storing events
- Data analytics for health statistics
- PyQt6-based user interface
- Dashboard with statistics and visualizations
- Settings panel for configuration
- System notifications for reminders
- Configuration management
- Logging system
- Documentation (user guide, developer guide, FAQ)
- Testing framework with pytest
- Pre-commit hooks for code quality
- CI/CD setup with GitHub Actions

### Enhanced
- Improved detection accuracy with configurable model complexity
- Added support for MediaPipe Heavy model for posture detection
- Added user interface controls to adjust model complexity
- Updated configuration system to store model complexity settings

## [0.1.0] - 2025-06-12

### Added
- Initial release of the Employee Health Monitoring System
- Core functionality for monitoring standing and drinking behaviors
- Basic UI with dashboard and settings
- Local data storage with SQLite
- System notifications for reminders
