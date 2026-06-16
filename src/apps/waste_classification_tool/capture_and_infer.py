#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
from capture_images_simple import main as capture_main
from runtime_paths import CAPTURE_DIR, OUTPUT_DIR, VOICE_RUNTIME_DIR, ensure_dirs

# 配置
CAPTURE_DIR = str(CAPTURE_DIR)
ONNX_SCRIPT = 'onnx_infer_output.py'
LOG_FILE_PATH = str(VOICE_RUNTIME_DIR / 'detection_log.txt')

def main():
    print("="*60)
    print("捕获图片")
    print("="*60)
    capture_main()      # 调用捕获图片的主函数

    print("\n" + "="*60)
    print("运行推理")
    print("="*60)
    ensure_dirs()

    # 临时修改环境变量（若推理脚本依赖这些变量，保留；否则可删除）
    env = os.environ.copy()
    env['ONNX_INPUT_DIR'] = CAPTURE_DIR
    env['ONNX_OUTPUT_DIR'] = str(OUTPUT_DIR)

    result = subprocess.run(['python3', ONNX_SCRIPT], env=env, cwd=os.path.dirname(__file__))

    if result.returncode == 0:
        print("\n推理完成！")
    else:
        print("\n推理失败！")
    try:
        src = LOG_FILE_PATH
        dest = os.path.join(os.path.dirname(src), "7.txt")
        if os.path.exists(src):
            os.replace(src, dest)
            print(f"已重命名: {src} -> {dest}")
        else:
            print(f"未找到日志文件: {src}")
    except Exception as e:
        print(f"重命名失败: {e}")
        
if __name__ == '__main__':
    main()
