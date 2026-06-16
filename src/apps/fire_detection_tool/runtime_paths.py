#!/usr/bin/env python3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captured_images"
TITLE_CROPS_DIR = BASE_DIR / "title_crops"
TEMP_TITLE_CROPS_DIR = BASE_DIR / "temp_title_crops"
ONNX_RESULTS_DIR = BASE_DIR / "onnx_results"
WEIGHTS_DIR = BASE_DIR / "weights"
OCR_DIR = BASE_DIR / "OCR"
VOICE_RUNTIME_DIR = BASE_DIR.parent / "voice_broadcast_tool" / "runtime" / "monitor_txt"


def ensure_dirs():
    for path in [
        CAPTURE_DIR,
        TITLE_CROPS_DIR,
        TEMP_TITLE_CROPS_DIR,
        ONNX_RESULTS_DIR,
        VOICE_RUNTIME_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
