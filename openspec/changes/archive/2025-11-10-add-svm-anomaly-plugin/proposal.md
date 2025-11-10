# Change: Add SVM Anomaly Detection Plugin

## Why
The project currently has pose detection and keypoints saving capabilities, but lacks anomaly detection to identify unusual poses that might indicate falls or other incidents. Adding SVM-based anomaly detection will enable real-time monitoring for pose anomalies using machine learning.

## What Changes
- Add new plugin `svm_anomaly_plugin` that uses One-Class SVM to detect pose anomalies
- Integrate with existing pose detection pipeline to analyze keypoints in real-time
- Provide configurable anomaly detection parameters (nu, kernel, gamma)
- Support loading pre-trained SVM models for consistent detection

## Impact
- Affected specs: anomaly-detection (new capability)
- Affected code: poseplay/plugins.py (plugin system), poseplay/lib/ (new plugin file)
- No breaking changes to existing functionality