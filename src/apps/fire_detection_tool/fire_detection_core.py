"""
火灾隐患检测核心模块
封装完整的检测流程
"""
import sys
from pathlib import Path
from runtime_paths import TEMP_TITLE_CROPS_DIR, WEIGHTS_DIR, OCR_DIR, ensure_dirs

# 添加模块路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "OCR"))

import onnx_infer_and_crop as yolo
from OCR import ocr_recognize

class FireDetectionCore:
    """火灾检测核心类"""
    
    def __init__(self, 
                 yolo_model_path=str(WEIGHTS_DIR / "best.onnx"),
                 ocr_model_path=str(OCR_DIR / "model.onnx"),
                 ocr_vocab_path=str(OCR_DIR / "vocab.txt")):
        """初始化模型"""
        self.yolo_session, self.yolo_inputs, self.input_w, self.input_h = \
            yolo.init_detect_model(yolo_model_path)
        self.ocr_model_path = ocr_model_path
        self.ocr_vocab_path = ocr_vocab_path
        ensure_dirs()
        self.temp_crop_dir = str(TEMP_TITLE_CROPS_DIR)
    
    def detect(self, image_path):
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            str: 格式 "xx大厦发现火灾隐患 N个"
        """
        # 1. YOLO 检测
        fire_count, title_crop_paths = yolo.detect_single_image(
            image_path, 
            self.yolo_session, 
            self.yolo_inputs, 
            self.input_w, 
            self.input_h,
            self.temp_crop_dir
        )
        
        # 2. OCR 识别
        building_name = "未知建筑"
        if title_crop_paths:
            building_name = ocr_recognize(
                title_crop_paths[0], 
                self.ocr_model_path, 
                self.ocr_vocab_path
            )
        
        # 3. 生成结果
        result = f"{building_name}发现火灾隐患 {fire_count}个"
        return result, fire_count, building_name

def detect_from_image(image_path):
    """
    便捷函数：从图像路径检测
    
    Args:
        image_path: 图像路径
    
    Returns:
        str: 检测结果
    """
    detector = FireDetectionCore()
    result, _, _ = detector.detect(image_path)
    return result
