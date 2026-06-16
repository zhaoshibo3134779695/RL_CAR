#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS摄像头实时识别 + ONNX目标检测 + 裁剪 + 文字识别 + 结果保存
"""
import rospy
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from onnxruntime import InferenceSession
import onnxruntime as rt
from runtime_paths import PLATE_MODEL, PLATE_OCR_MODEL, PLATE_VOCAB, RUNTIME_PLATE, VOICE_RUNTIME_DIR, ensure_runtime_dirs


# ========== 核心配置区域 ==========
# ROS摄像头配置
CAMERA_TOPIC = '/camera/color/image_raw'  # 相机话题
CAPTURE_NUM = 1                             # 要捕获并处理的图片数量

# 保存配置（修改为cp文件夹）
OUTPUT_DIR = str(RUNTIME_PLATE)
START_FILE_NUM = 5  # 从5开始命名文件（5.txt、6.txt...）

# 指定.txt文件的保存目录
TXT_OUTPUT_DIR = str(VOICE_RUNTIME_DIR)

# 目标检测配置
DET_MODEL_PATH = str(PLATE_MODEL)
LABEL_MAP = {0: "chepai"}                   # 类别映射
INPUT_SIZE = (640, 640)                                   # 模型输入尺寸 (width, height)
CONF_THRESHOLD = 0.5                                      # 置信度阈值
IOU_THRESHOLD = 0.2                                       # NMS阈值
NUM_CLASSES = 2                                           # 类别数量

# 裁剪区域百分比配置（保持原有）
CROP_LEFT_PERCENT = 34  # 框内左侧起始百分比
CROP_RIGHT_PERCENT = 67  # 框内右侧结束百分比
CROP_TOP_PERCENT = 42  # 框内顶部起始百分比
CROP_BOTTOM_PERCENT = 58  # 框内底部结束百分比

# 文字识别相关配置（保持原有）
OCR_MODEL_PATH = str(PLATE_OCR_MODEL)
VOCAB_PATH = str(PLATE_VOCAB)
DEVICE = "cpu"  # 改为"cuda"如果有GPU
# ==================================


# 全局变量初始化
bridge = CvBridge()
count = 0
saved_files = []
ensure_runtime_dirs()


# -------------------------- 工具函数：获取递增文件编号 --------------------------
def get_next_file_num(save_dir, ext=".txt"):
    """
    获取下一个递增的文件编号，避免覆盖
    :param save_dir: 保存目录
    :param ext: 文件扩展名（用于匹配已有文件）
    :return: 下一个可用的数字编号
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 匹配所有 数字+扩展名 的文件（如5.txt、6.txt）
    pattern = os.path.join(save_dir, f"*{ext}")
    files = glob.glob(pattern)
    
    max_num = START_FILE_NUM - 1
    for f in files:
        # 提取文件名中的数字部分
        fname = os.path.splitext(os.path.basename(f))[0]
        if fname.isdigit():
            num = int(fname)
            if num > max_num:
                max_num = num
    return max_num + 1


# -------------------------- 初始化设置 --------------------------
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)  # 增加的.txt保存目录
print(f"=== Output Directory: {OUTPUT_DIR} ===")
print(f"重要：请打开 {OUTPUT_DIR} 文件夹，查看裁剪图、标注检测图和识别结果TXT！\n")

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

# 加载文字识别ONNX模型（全局加载，避免重复加载）
try:
    ocr_session = rt.InferenceSession(OCR_MODEL_PATH)
    ocr_input_name = ocr_session.get_inputs()[0].name
    ocr_output_name = ocr_session.get_outputs()[0].name
    print(f"Successfully loaded OCR model: {OCR_MODEL_PATH}\n")
except Exception as e:
    print(f"Failed to load OCR model: {e}")
    exit(1)

# 加载文字识别字典
try:
    labelMapping = dict()
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        cnt = 2
        for line in lines:
            line = line.strip('\n')
            labelMapping[cnt] = line
            cnt += 1
    print(f"Successfully loaded vocab: {VOCAB_PATH}\n")
except Exception as e:
    print(f"Failed to load vocab: {e}")
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

    # 1. 调整维度（修正注释，实际是5维）
    pred = pred.transpose(0, 2, 1)[0]  # (8400, 5)
    print(f"  筛选前检测框数量: {len(pred)}")
    print(f"  检测框特征维度: {pred.shape[1]}")  # 新增：打印维度，方便排查

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

    # 6. NMS去重（核心修正：处理类别 + 鲁棒性优化）
    confs = pred[:, 4]
    # 修正1：模型无类别列，单类别场景下固定类别为0（可根据实际调整）
    clss = np.zeros_like(confs, dtype=int)  # 所有检测框类别为0
    
    # 修正2：处理NMS返回标量的情况（仅1个框时）
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


# 2. 文字识别相关工具函数
def keepratio_resize(img):
    """保持比例缩放（文字识别专用）"""
    cur_ratio = img.shape[1] / float(img.shape[0])
    mask_height = 32
    mask_width = 804
    if cur_ratio > float(mask_width) / mask_height:
        cur_target_height = mask_height
        cur_target_width = mask_width
    else:
        cur_target_height = mask_height
        cur_target_width = int(mask_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
    mask[:img.shape[0], :img.shape[1], :] = img
    img = mask
    return img


def ocr_recognize(img_path):
    """文字识别核心函数：纯NumPy实现，彻底移除PyTorch依赖"""
    try:
        # 1. 读取并预处理图片
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        img = keepratio_resize(img)
        img_np = img.astype(np.float32) / 255.0  # 归一化到0-1

        # 2. 切分并拼接图片（纯NumPy实现）
        chunk_img = []
        for i in range(3):
            left = (300 - 48) * i
            # NumPy切片替代torch切片
            chunk = img_np[:, left:left + 300, :]
            chunk_img.append(chunk)
        # NumPy拼接替代torch.cat
        merge_img = np.concatenate(chunk_img, axis=0)
        # 重塑形状（匹配原逻辑：3, 32, 300, 3）
        merge_img = merge_img.reshape(3, 32, 300, 3)
        # 调整维度顺序（替代torch.permute）: (batch, h, w, c) → (batch, c, h, w)
        data = np.transpose(merge_img, (0, 3, 1, 2))

        # 3. ONNX模型推理（纯NumPy输入，无需转torch）
        res = ocr_session.run([ocr_output_name], {ocr_input_name: data})
        outprobs = res[0]

        # 4. 纯NumPy实现Softmax + 解码（替代torch的softmax/argmax）
        def softmax(x, axis=-1):
            """纯NumPy实现Softmax，防止数值溢出"""
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        # 计算Softmax概率
        outprobs = softmax(outprobs)
        # 纯NumPy找最大值索引（替代torch.argmax）
        preds = np.argmax(outprobs, axis=-1)

        # 5. 解码识别结果（纯Python/NumPy）
        batchSize, length = preds.shape
        final_str_list = []
        for i in range(batchSize):
            pred_idx = preds[i].tolist()  # NumPy转列表
            last_p = 0
            str_pred = []
            for p in pred_idx:
                if p != last_p and p != 0:
                    str_pred.append(labelMapping.get(p, ""))
                last_p = p
            final_str = ''.join(str_pred)
            final_str_list.append(final_str)

        # 合并结果（去除空字符串）
        final_text = ''.join([s for s in final_str_list if s])
        return final_text.strip()

    except Exception as e:
        print(f"文字识别失败: {e}")
        # 打印详细报错栈，方便定位（可选）
        import traceback
        traceback.print_exc()
        return ""


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

        # 5. 处理检测结果并裁剪+文字识别
        crop_success = False
        if detection_results:
            # 获取下一个文件编号（从5开始，避免覆盖）
            file_num = get_next_file_num(OUTPUT_DIR)
            
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

                # 计算裁剪区域坐标
                box_w = x2 - x1
                box_h = y2 - y1
                crop_x1 = x1 + int(box_w * CROP_LEFT_PERCENT / 100)
                crop_y1 = y1 + int(box_h * CROP_TOP_PERCENT / 100)
                crop_x2 = x1 + int(box_w * CROP_RIGHT_PERCENT / 100)
                crop_y2 = y1 + int(box_h * CROP_BOTTOM_PERCENT / 100)

                # 确保裁剪坐标在图像范围内
                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(w0, crop_x2)
                crop_y2 = min(h0, crop_y2)

                # 裁剪并保存
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    cropped_img = cv_img[crop_y1:crop_y2, crop_x1:crop_x2]
                    # 裁剪图命名：crop_5.jpg、crop_6.jpg...
                    crop_img_name = f"crop_{file_num}.jpg"
                    crop_img_path = os.path.join(OUTPUT_DIR, crop_img_name)
                    cv2.imwrite(crop_img_path, cropped_img)
                    crop_success = True
                    print(f"  裁剪区域坐标: ({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})")
                    print(f"  裁剪图已保存到: {crop_img_path}")
                    print(f"  裁剪图大小: {os.path.getsize(crop_img_path) / 1024:.2f} KB")

                    # 文字识别
                    ocr_text = ocr_recognize(crop_img_path)
                    print(f"  文字识别结果: {ocr_text if ocr_text else '无识别结果'}")

                    # 保存识别结果：5.txt、6.txt...（内容格式不变）
                    txt_name = "cp_2.txt"
                    txt_path = os.path.join(TXT_OUTPUT_DIR, txt_name)
                    # 指定.txt存储内容格式
                    content = f"二号停车场车牌号为:{ocr_text}。" if ocr_text else "二号停车场车牌号识别失败"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  识别结果已保存到: {txt_path}，内容：{content}")

                else:
                    print(f"  裁剪区域无效：宽={crop_x2 - crop_x1}, 高={crop_y2 - crop_y1}")

            # 保存标注检测图：detected_5.jpg、detected_6.jpg...
            detected_img_name = f"detected_{file_num}.jpg"
            detected_img_path = os.path.join(OUTPUT_DIR, detected_img_name)
            cv2.imwrite(detected_img_path, cv_img)
            print(f"  标注置信度的检测图片已保存至: {detected_img_path}")
            print(f"  模型检测完成")
        
        else:
            print(f"  无模型检测框，跳过裁剪和识别")
            file_num = get_next_file_num(TXT_OUTPUT_DIR)
            txt_name = "cp_2.txt"
            txt_path = os.path.join(TXT_OUTPUT_DIR, txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("二号停车场车牌号识别失败")
            print(f"  识别结果已保存到: {txt_path}，内容：二号停车场车牌号识别失败")

        # 更新计数和保存列表
        count += 1
        if crop_success:
            saved_files.append({
                'crop_img': crop_img_path,
                'ocr_txt': txt_path if 'txt_path' in locals() else None,
                'detected_img': detected_img_path if 'detected_img_path' in locals() else None
            })
        
    except Exception as e:
        print(f"第{count + 1}张图处理失败: {str(e)}")


# -------------------------- 主函数 --------------------------
def main():
    # 初始化ROS节点
    rospy.init_node('ros_camera_onnx_ocr_detection', anonymous=True)
    
    print("=" * 60)
    print("ROS摄像头实时识别+文字识别+结果保存系统")
    print("=" * 60)
    print(f"摄像头话题: {CAMERA_TOPIC}")
    print(f"目标捕获数量: {CAPTURE_NUM}")
    print(f"保存目录: {OUTPUT_DIR}")
    print(f"检测模型路径: {DET_MODEL_PATH}")
    print(f"OCR模型路径: {OCR_MODEL_PATH}")
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
        print(f"  {i}. 裁剪图: {item['crop_img']}")
        if item['ocr_txt']:
            print(f"     识别结果: {item['ocr_txt']}")
        if item['detected_img']:
            print(f"     标注检测图: {item['detected_img']}")
    print(f"请打开输出文件夹查看完整结果：{OUTPUT_DIR}")
    print(f"裁剪参数：左右[{CROP_LEFT_PERCENT}%-{CROP_RIGHT_PERCENT}%]，上下[{CROP_TOP_PERCENT}%-{CROP_BOTTOM_PERCENT}%]")
    print("=" * 60)
    try:
        old_txt_path = os.path.join(TXT_OUTPUT_DIR, "cp_2.txt")
        new_txt_path = os.path.join(TXT_OUTPUT_DIR, "6.txt")
        if os.path.exists(old_txt_path):
            # 如需覆盖已存在的 6.txt，可先删除或使用 os.replace
            if os.path.exists(new_txt_path):
                os.remove(new_txt_path)
            os.rename(old_txt_path, new_txt_path)
            print(f"已重命名文本文件: {old_txt_path} → {new_txt_path}")
        else:
            print(f"未找到待重命名的文件: {old_txt_path}")
    except Exception as e:
        print(f"重命名失败: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n 收到中断信号，程序退出")
    except Exception as e:
        print(f"\n\n 程序异常退出: {str(e)}")
