#!/usr/bin/env python3
"""
火灾隐患检测流程 - 主调用文件
功能：ROS 捕获图像 → YOLO 检测 → OCR 识别 → 输出结果
"""
from pathlib import Path
from fire_detection_core import FireDetectionCore
from capture_images_simple import main as capture_images
import onnx_infer_and_crop as yolo
import os
from runtime_paths import CAPTURE_DIR, WEIGHTS_DIR, OCR_DIR, ONNX_RESULTS_DIR, TITLE_CROPS_DIR, VOICE_RUNTIME_DIR, ensure_dirs

# ==================== 配置项 ====================
# 路径配置
CAPTURE_DIR = str(CAPTURE_DIR)
YOLO_MODEL = str(WEIGHTS_DIR / 'best.onnx')
OCR_MODEL = str(OCR_DIR / 'model.onnx')
OCR_VOCAB = str(OCR_DIR / 'vocab.txt')
RESULT_FILE = str(VOICE_RUNTIME_DIR / 'fire_detection_result2.txt')
# ================================================

def init_environment():
    """初始化运行环境"""
    ensure_dirs()
    required_files = [YOLO_MODEL, OCR_MODEL, OCR_VOCAB]
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        error_msg = f"以下关键文件缺失：{missing_files}"
        print(f"✗ {error_msg}")
        raise FileNotFoundError(error_msg)
    print("✓ 运行环境初始化完成")

def main():
    """主流程控制"""
    try:
        init_environment()
    except Exception as e:
        print(f"✗ 环境初始化失败：{e}")
        return

    print("\n" + "=" * 60)
    print("步骤 1/4: 捕获图像")
    print("=" * 60)

    # 1. 捕获图像
    try:
        capture_images()
    except Exception as e:
        print(f"✗ 图像捕获失败：{e}")
        return

    print("\n" + "=" * 60)
    print("步骤 2/4: 检测与识别")
    print("=" * 60)

    # 2. 找到最新的捕获图像
    capture_path = Path(CAPTURE_DIR)
    images = sorted(capture_path.glob('*.jpg'))
    if not images:
        print("✗ 未找到捕获的图像，请检查图像捕获功能")
        return

    latest_image = str(images[-1])
    print(f"✓ 找到最新图像: {Path(latest_image).name}")
    # 新增：生成 onnx_results（带检测框的图像）
    print("\n生成检测框可视化结果...")
    OUTPUT_DIR = str(ONNX_RESULTS_DIR)
    CROP_OUTPUT_DIR = str(TITLE_CROPS_DIR)
    detector_visual = yolo.init_detect_model(YOLO_MODEL)
    yolo.infer_folder(CAPTURE_DIR, OUTPUT_DIR, detector_visual[0], detector_visual[1], 
                      detector_visual[2], detector_visual[3], crop_output_dir=CROP_OUTPUT_DIR)
    print(f"✓ 可视化结果已保存到 {OUTPUT_DIR}")
    
    
    # 3. 初始化检测器并执行检测
    try:
        detector = FireDetectionCore(YOLO_MODEL, OCR_MODEL, OCR_VOCAB)
        result, fire_count, building_name = detector.detect(latest_image)
    except Exception as e:
        print(f"✗ 检测识别失败：{e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("步骤 3/4: 输出结果")
    print("=" * 60)
    print(f"\n 检测结果汇总：")
    print(f"   火灾隐患数量：{fire_count} 个")
    print(f"   识别建筑名称：{building_name if building_name else '未知'}")
    print(f"   最终结论：{result}\n")

    # 4. 保存结果
    try:
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{building_name if building_name else '未知'}发现火灾隐患{fire_count}个\n")
        print(f"✓ 结果已保存到 {RESULT_FILE}")
    except Exception as e:
        print(f"✗ 保存结果失败：{e}")

    print("\n" + "=" * 60)
    print(" 火灾隐患检测流程完成")
    print("=" * 60)
    try:
        src = RESULT_FILE
        dest = os.path.join(os.path.dirname(src), "9.txt")
        if os.path.exists(src):
            os.replace(src, dest)
            print(f"✓ 已重命名: {src} -> {dest}")
        else:
            print(f"✗ 未找到结果文件: {src}")
    except Exception as e:
        print(f"✗ 重命名失败：{e}")
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ 火灾隐患检测流程被用户手动中断")
    except Exception as e:
        print(f"\n✗ 程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()
