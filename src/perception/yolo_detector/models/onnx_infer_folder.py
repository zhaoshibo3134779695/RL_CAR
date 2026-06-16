#!/usr/bin/env python3
import os
from ultralytics import YOLO

# ===== 1. 配置路径 =====
onnx_path = "best.onnx"           # 你的 onnx 模型路径
input_dir = "input_images"            # 要检测的图片所在文件夹
output_dir = "onnx_output_results"          # 检测后保存图片的文件夹

# ===== 2. 加载 ONNX 模型 =====
model = YOLO(onnx_path)  # 这里加载的是 onnx，不是 pt

# ===== 3. 创建输出文件夹 =====
os.makedirs(output_dir, exist_ok=True)

# ===== 4. 遍历输入文件夹里的所有图片 =====
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for name in os.listdir(input_dir):
    if not name.lower().endswith(valid_ext):
        continue

    img_path = os.path.join(input_dir, name)
    print(f"正在检测: {img_path}")

    # 推理，返回的 results 列表，这里只处理第一张
    results = model(img_path, save=False)[0]

    # 用自带的画框函数生成带框图像（numpy 数组，BGR）
    im = results.plot()

    # 保存到输出文件夹
    save_path = os.path.join(output_dir, name)
    import cv2
    cv2.imwrite(save_path, im)

    print(f"已保存: {save_path}")

print("全部图片检测完成。")
