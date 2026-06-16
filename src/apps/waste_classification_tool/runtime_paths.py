#!/usr/bin/env python3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captured_images"
OUTPUT_DIR = BASE_DIR / "onnx_classification_results"
WEIGHTS_DIR = BASE_DIR / "weights"
VOICE_RUNTIME_DIR = BASE_DIR.parent / "voice_broadcast_tool" / "runtime" / "monitor_txt"


def ensure_dirs():
    for path in [CAPTURE_DIR, OUTPUT_DIR, VOICE_RUNTIME_DIR]:
        path.mkdir(parents=True, exist_ok=True)
