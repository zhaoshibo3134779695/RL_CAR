# Upload Notes

This repository intentionally excludes a few large runtime assets so it can be pushed to GitHub without Git LFS.

Excluded files:

- `src/apps/waste_classification_tool/weights/best_m.onnx`
- `src/perception/yolo_detector/models/people/renwu.onnx`
- `src/apps/voice_broadcast_tool/piper`
- `src/apps/voice_broadcast_tool/lib*.so*`
- `src/apps/voice_broadcast_tool/espeak-ng-data/`
- `src/apps/voice_broadcast_tool/zh_CN-huayan-medium.onnx`
- `src/imu_tools/imu_filter_madgwick/sample/*.bag`

Reason:

- GitHub rejects individual files over 100 MB.
- The voice broadcast runtime bundle is platform-specific binary payload, not source.
- The excluded `.bag` files are sample data rather than required source code.

Everything else that remains in the workspace was kept so the repository still contains the ROS package source, launch files, maps, URDF, and the smaller model assets currently used by the project.
