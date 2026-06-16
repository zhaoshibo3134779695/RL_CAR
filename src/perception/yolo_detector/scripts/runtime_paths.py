#!/usr/bin/env python3
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_SRC_DIR = PACKAGE_DIR.parent.parent

MODELS_DIR = PACKAGE_DIR / "models"
RUNTIME_DIR = PACKAGE_DIR / "runtime"

PEOPLE_MODEL = MODELS_DIR / "people" / "renwu.onnx"
PARKING_MODEL = MODELS_DIR / "parking" / "weiting.onnx"
TRAFFIC_SIGN_MODEL = MODELS_DIR / "traffic" / "bz.onnx"
PLATE_MODEL = MODELS_DIR / "plate" / "chepai.onnx"
PLATE_OCR_MODEL = MODELS_DIR / "ocr" / "model.onnx"
PLATE_VOCAB = MODELS_DIR / "ocr" / "vocab.txt"
EBIKE_MODEL = MODELS_DIR / "ebike" / "ddc.onnx"

RUNTIME_R = RUNTIME_DIR / "r"
RUNTIME_PARKING = RUNTIME_DIR / "weiting"
RUNTIME_DDC_TXT = RUNTIME_DIR / "ddc_txt"
RUNTIME_DDC_IMG = RUNTIME_DIR / "ddctp"
RUNTIME_SIGN = RUNTIME_DIR / "bz"
RUNTIME_PLATE = RUNTIME_DIR / "cp"

VOICE_RUNTIME_DIR = WORKSPACE_SRC_DIR / "apps" / "voice_broadcast_tool" / "runtime" / "monitor_txt"


def ensure_runtime_dirs():
    for path in [
        RUNTIME_DIR,
        RUNTIME_R,
        RUNTIME_PARKING,
        RUNTIME_DDC_TXT,
        RUNTIME_DDC_IMG,
        RUNTIME_SIGN,
        RUNTIME_PLATE,
        VOICE_RUNTIME_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
