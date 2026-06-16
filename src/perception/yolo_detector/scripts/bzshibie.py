#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS摄像头实时识别 + ONNX目标检测 + 结果保存
修改后：仅检测"bz"标签，保存标注后的原图和1.txt（固定内容）
"""
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from onnxruntime import InferenceSession
from runtime_paths import TRAFFIC_SIGN_MODEL, RUNTIME_SIGN, VOICE_RUNTIME_DIR, ensure_runtime_dirs


# ========== 核心配置区域 ==========
# ROS摄像头配置
CAMERA_TOPIC = '/camera/color/image_raw'  # 相机话题
CAPTURE_NUM = 1                             # 要捕获并处理的图片数量

# 保存配置（固定保存为1.txt、detected_1.jpg）
OUTPUT_DIR = str(RUNTIME_SIGN)
# 指定.txt文件输出路径
TXT_OUTPUT_DIR = str(VOICE_RUNTIME_DIR)
# 指定.txt文件名
TXT_FILE_NAME = "biaozhi.txt"

# 目标检测配置
DET_MODEL_PATH = str(TRAFFIC_SIGN_MODEL)
LABEL_MAP = {0: "bz"}                       # 类别映射：0对应"bz"
INPUT_SIZE = (640, 640)                     # 模型输入尺寸 (width, height)
CONF_THRESHOLD = 0.5                        # 置信度阈值
IOU_THRESHOLD = 0.2                         # NMS阈值
NUM_CLASSES = 1                             # 类别数量（仅bz）
# ==================================


# 全局变量初始化
bridge = CvBridge()
count = 0
saved_files = []
ensure_runtime_dirs()


# -------------------------- 初始化设置 --------------------------
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
print(f"=== Output Directory: {OUTPUT_DIR} ===")
print(f"⚠️  重要：请打开 {OUTPUT_DIR} 文件夹，查看标注检测图和1.txt！\n")

# 加载目标检测ONNX模型
try:
    det_session = InferenceSession(
        DET_MODEL_PATH,
        providers=['CPUExecutionProvider']
    )
    det_input_name = det_session.get_inputs()[0].name
    det_output_names = [out.name for out in det_session.get_outputs()]
    det_input_shape = det_session.get_inputs()[0].shape
    print(f"Detection Model Input Shape: {det_input_shape} (batch, channel, height, width)")
    print(f"Successfully loaded detection model: {DET_MODEL_PATH}\n")
except Exception as e:
    print(f"Failed to load detection model: {e}")
    exit(1)


# -------------------------- 工具函数 --------------------------
# 1. 目标检测相关工具函数
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """保持比例缩放 + 边缘填充（YOLOv8标准）"""
    h0, w0 = img.shape[:2]  # 原始图：(height, width)
    target_h, target_w = new_shape[1], new_shape[0]  # 目标：(height, width)

    # 计算缩放比例
    r = min(target_h / h0, target_w / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)

    # 计算填充量
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2

    # 缩放 + 填充
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, pad_h, target_h - new_h - pad_h,
                             pad_w, target_w - new_w - pad_w,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (pad_w, pad_h)


def preprocess_image(img):
    """预处理：对齐YOLOv8"""
    try:
        img_processed, r, pad = letterbox(img, new_shape=INPUT_SIZE)
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        img_processed = img_processed / 255.0
        img_processed = np.transpose(img_processed, (2, 0, 1))  # (h,w,c)→(c,h,w)
        img_processed = np.expand_dims(img_processed, axis=0)
        return img_processed.astype(np.float32), r, pad
    except Exception as e:
        print(f"Preprocess failed: {e}")
        return None, None, None


def postprocess_output(pred, img_original_shape, r, pad, conf_thres=0.1, iou_thres=0.45):
    """后处理：修复坐标计算 + 解决索引越界 + 鲁棒性优化"""
    if pred is None or r is None or pad is None:
        return []

    h0, w0 = img_original_shape  # 原始图：(height, width)
    pad_w, pad_h = pad

    # 1. 调整维度
    pred = pred.transpose(0, 2, 1)[0]  # (8400, 5)
    print(f"  筛选前检测框数量: {len(pred)}")
    print(f"  检测框特征维度: {pred.shape[1]}")

    # 2. 筛选高置信度框
    conf_mask = pred[:, 4] > conf_thres
    pred = pred[conf_mask]
    print(f"  置信度筛选后框数量: {len(pred)}")
    if len(pred) == 0:
        return []

    # 3. 解析xywh（模型输出是640x640图中的绝对像素值）
    x_center = pred[:, 0]
    y_center = pred[:, 1]
    w = pred[:, 2]
    h = pred[:, 3]
    print(f"  模型输出xywh（640x640图）: x={x_center[0]:.0f}, y={y_center[0]:.0f}, w={w[0]:.0f}, h={h[0]:.0f}")

    # 转换为x1y1x2y2
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    print(f"  转换后x1y1x2y2（640x640图）: {x1[0]:.0f}, {y1[0]:.0f}, {x2[0]:.0f}, {y2[0]:.0f}")

    # 4. 坐标还原：减去填充 + 除以缩放比例
    x1 -= pad_w
    y1 -= pad_h
    x2 -= pad_w
    y2 -= pad_h
    print(f"  减去填充后坐标（640x640图）: {x1[0]:.0f}, {y1[0]:.0f}, {x2[0]:.0f}, {y2[0]:.0f}")

    x1 /= r
    y1 /= r
    x2 /= r
    y2 /= r
    print(f"  还原到原始图坐标: {x1[0]:.0f}, {y1[0]:.0f}, {x2[0]:.0f}, {y2[0]:.0f}")

    # 5. 限制坐标范围
    boxes = np.stack((x1, y1, x2, y2), axis=1)
    boxes = np.clip(boxes, 0, [w0, h0, w0, h0])
    print(f"  限制后最终坐标: {boxes[0][0]:.0f}, {boxes[0][1]:.0f}, {boxes[0][2]:.0f}, {boxes[0][3]:.0f}")

    # 6. NMS去重
    confs = pred[:, 4]
    clss = np.zeros_like(confs, dtype=int)  # 所有检测框类别为0（bz）
    
    # 处理NMS返回标量的情况（仅1个框时）
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), conf_thres, iou_thres)
    if isinstance(indices, (int, np.integer)):  # 标量转数组
        indices = np.array([indices])
    print(f"  NMS后保留框数量: {len(indices)}")

    # 7. 整理结果
    results = []
    for i in indices.flatten():
        x1, y1, x2, y2 = map(int, boxes[i])
        results.append((x1, y1, x2, y2, round(confs[i], 4), clss[i]))
    return results


# -------------------------- ROS回调函数（核心处理逻辑） --------------------------
def image_callback(msg):
    global count, saved_files
    
    # 达到捕获数量后停止处理
    if count >= CAPTURE_NUM:
        return
    
    try:
        # 1. 从ROS消息转换为OpenCV图像
        cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        h0, w0 = cv_img.shape[:2]
        print(f"\n[{count + 1}/{CAPTURE_NUM}] 收到相机图像 - 尺寸: height={h0}, width={w0}")
        
        # 2. 预处理（目标检测）
        img_input, r, pad = preprocess_image(cv_img)
        if img_input is None:
            raise RuntimeError("Preprocess failed")
        print(f"  缩放比例: {r:.4f}, 填充量: (pad_w={pad[0]}, pad_h={pad[1]})")

        # 3. 目标检测模型推理
        preds = det_session.run(det_output_names, {det_input_name: img_input})
        pred_boxes = preds[0]
        print(f"  模型输出形状: {pred_boxes.shape}")

        # 4. 后处理（修复坐标）
        detection_results = postprocess_output(pred_boxes, (h0, w0), r, pad, CONF_THRESHOLD, IOU_THRESHOLD)

        # 5. 处理检测结果并保存（仅保存标注后的原图和1.txt）
        save_success = False
        file_num = 1  # 固定文件编号为1
        
        if detection_results:
            for (x1, y1, x2, y2, conf, cls) in detection_results:
                label = LABEL_MAP.get(cls, f"cls_{cls}")
                text = f"{label} ({conf:.2f})"
                print(f"  检测到目标: {text}")

                # 绘制检测框和置信度标签
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                label_text = f"{label} {conf:.2f}"
                cv2.putText(cv_img, label_text, (x1, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 保存标注检测图（固定命名为detected_1.jpg）
            detected_img_name = f"detected_{file_num}.jpg"
            detected_img_path = os.path.join(OUTPUT_DIR, detected_img_name)
            cv2.imwrite(detected_img_path, cv_img)
            print(f"  标注置信度的检测图片已保存至: {detected_img_path}")
            
            # 保存.txt（固定内容：前方是禁止直行指示牌）
            txt_name = TXT_FILE_NAME
            txt_path = os.path.join(TXT_OUTPUT_DIR, txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("前方是禁止直行指示牌")
            print(f"  结果文本已保存到: {txt_path} (内容：前方是禁止直行指示牌)")
            save_success = True
            print(f"  模型检测完成")
        
        else:
            print(f"  无模型检测框，仅保存原图（无标注）")
            # 无检测框时也保存原图（可选，如需取消可删除以下3行）
            detected_img_name = f"detected_{file_num}.jpg"
            detected_img_path = os.path.join(OUTPUT_DIR, detected_img_name)
            cv2.imwrite(detected_img_path, cv_img)
            print(f"  原图已保存至: {detected_img_path}")
            
            # 保存.txt（即使无检测框也保存）
            txt_name = TXT_FILE_NAME
            txt_path = os.path.join(TXT_OUTPUT_DIR, txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("前方是禁止直行指示牌")
            print(f"  结果文本已保存到: {txt_path} (内容：前方是禁止直行指示牌)")
        
        # 更新计数和保存列表
        count += 1
        if save_success or 'detected_img_path' in locals():
            saved_files.append({
                'detected_img': detected_img_path,
                'result_txt': txt_path if 'txt_path' in locals() else None
            })
        
    except Exception as e:
        print(f"❌ 第{count + 1}张图处理失败: {str(e)}")


# -------------------------- 主函数 --------------------------
def main():
    # 初始化ROS节点
    rospy.init_node('ros_camera_onnx_bz_detection', anonymous=True)
    
    print("=" * 60)
    print("ROS摄像头实时识别 + bz目标检测 + 固定结果保存")
    print("=" * 60)
    print(f"摄像头话题: {CAMERA_TOPIC}")
    print(f"目标捕获数量: {CAPTURE_NUM}")
    print(f"保存目录: {OUTPUT_DIR}")
    print(f"检测模型路径: {DET_MODEL_PATH}")
    print(f"检测标签: 0 → 'bz'")
    print("保存文件: detected_1.jpg（标注原图）、1.txt（固定内容）")
    print("=" * 60)
    print("开始接收图像...（按Ctrl+C停止）")
    
    # 订阅摄像头话题
    sub = rospy.Subscriber(CAMERA_TOPIC, Image, image_callback, queue_size=1)
    
    # 循环等待
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and count < CAPTURE_NUM:
        rate.sleep()
    
    # 取消订阅
    sub.unregister()
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"共处理 {count} 张图像，保存结果如下:")
    for i, item in enumerate(saved_files, 1):
        print(f"  {i}. 标注原图: {item['detected_img']}")
        if item['result_txt']:
            print(f"     结果文本: {item['result_txt']} (内容：前方是禁止直行指示牌)")
    
    # 重命名.txt文件为1.txt（如果不是的话）
    if TXT_FILE_NAME != "1.txt":
        old_txt_path = os.path.join(TXT_OUTPUT_DIR, TXT_FILE_NAME)
        new_txt_path = os.path.join(TXT_OUTPUT_DIR, "1.txt")
        try:
            if os.path.exists(old_txt_path):
                os.rename(old_txt_path, new_txt_path)
                print(f"  文件重命名成功: {TXT_FILE_NAME} → 1.txt")
                print(f"  新路径: {new_txt_path}")
            else:
                print(f"源文件不存在: {old_txt_path}")
        except Exception as e:
            print(f"文件重命名失败: {str(e)}")
    
    print(f"检测结果输出文件夹{OUTPUT_DIR}")
    print(f"检测结果文本文件夹{TXT_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n收到中断信号，程序退出")
    except Exception as e:
        print(f"\n\n程序异常退出: {str(e)}")
