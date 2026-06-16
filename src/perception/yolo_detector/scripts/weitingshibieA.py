#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS摄像头话题识别 + ONNX目标检测 + 累加计数保存
最终版：对齐ROS摄像头代码参数（消除色差）+ 避免过曝 + 调用camera/image_raw话题
修改：新增百分比可调的上半区域检测（默认上三分之二）+ 绘制识别区域分割线
"""
import cv2
import numpy as np
import os
import time
import re
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from onnxruntime import InferenceSession
from runtime_paths import PARKING_MODEL, RUNTIME_PARKING, RUNTIME_DDC_TXT, ensure_runtime_dirs


# ========== 核心配置区域（对齐ROS代码） ==========
# ROS摄像头配置（替换原CAMERA_INDEX）
CAMERA_TOPIC = "camera/image_raw"  # ROS摄像头话题名
CAMERA_TIMEOUT = 5.0  # 话题获取图像超时时间（秒）
CAPTURE_NUM = 1  # 固定读取1帧

# 保存配置
OUTPUT_DIR = str(RUNTIME_PARKING)
COUNT_TXT_NAME = "weiting_A.txt"
TXT_OUTPUT_DIR = str(RUNTIME_DDC_TXT)

# 目标检测配置
DET_MODEL_PATH = str(PARKING_MODEL)
LABEL_MAP = {0: "ddc"}  
INPUT_SIZE = (640, 640)  
CONF_THRESHOLD = 0.2  
IOU_THRESHOLD = 0.2  
NUM_CLASSES = 1  

# 摄像头参数（完全对齐ROS代码，避免色差+过曝）
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# 检测区域配置（百分比，便于调整）
DETECTION_AREA_TOP_PERCENT = 60  # 只检测图像上66.67%区域（三分之二），可修改为70/80/90等
# ===============================================

# ROS全局变量初始化
bridge = CvBridge()
received_frame = None
frame_received = False
ensure_runtime_dirs()


# -------------------------- ROS图像回调函数 --------------------------
def image_callback(msg):
    """ROS图像话题回调，接收图像并存储"""
    global received_frame, frame_received
    try:
        # 将ROS Image消息转换为OpenCV格式（BGR8）
        received_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        frame_received = True
    except Exception as e:
        rospy.logerr(f"ROS图像转换失败: {str(e)}")


# -------------------------- 工具函数 --------------------------
def get_ros_camera_frame(topic_name, timeout=5.0):
    """
    从ROS话题获取摄像头帧（替代原本地摄像头读取函数）
    :param topic_name: ROS图像话题名
    :param timeout: 超时时间（秒）
    :return: OpenCV图像帧 / None
    """
    global received_frame, frame_received
    # 重置全局变量
    received_frame = None
    frame_received = False

    # 订阅ROS图像话题
    sub = rospy.Subscriber(topic_name, Image, image_callback, queue_size=1)
    
    # 等待接收图像（超时退出）
    start_time = rospy.Time.now()
    while not rospy.is_shutdown() and not frame_received:
        if (rospy.Time.now() - start_time).to_sec() > timeout:
            rospy.logerr(f"获取图像超时！{timeout}秒内未从{topic_name}收到图像")
            sub.unregister()
            return None
        rospy.sleep(0.05)  # 50ms轮询一次
    
    # 取消订阅（避免重复接收）
    sub.unregister()
    
    if received_frame is not None:
        rospy.loginfo(f"成功获取ROS图像，尺寸: {received_frame.shape[:2]} (height, width)")
        return received_frame
    else:
        rospy.logerr("未获取到有效图像")
        return None


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """保持比例缩放 + 边缘填充（YOLOv8标准）"""
    h0, w0 = img.shape[:2]  
    target_h, target_w = new_shape[1], new_shape[0]  

    r = min(target_h / h0, target_w / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)

    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2

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
        img_processed = np.transpose(img_processed, (2, 0, 1))  
        img_processed = np.expand_dims(img_processed, axis=0)
        return img_processed.astype(np.float32), r, pad
    except Exception as e:
        print("Preprocess failed: {}".format(e))
        return None, None, None


def postprocess_output(pred, img_original_shape, r, pad, conf_thres=0.1, iou_thres=0.45):
    """后处理：修复坐标计算 + 解决索引越界 + 鲁棒性优化 + 检测区域过滤"""
    if pred is None or r is None or pad is None:
        return []

    h0, w0 = img_original_shape  
    pad_w, pad_h = pad

    # 转置并取第一个维度
    pred = pred.transpose(0, 2, 1)[0]  
    print("  筛选前检测框数量: {}".format(len(pred)))
    print("  检测框特征维度: {}".format(pred.shape[1]))

    # 1. 置信度筛选
    conf_mask = pred[:, 4] > conf_thres
    pred_filtered_by_conf = pred[conf_mask]
    print("  置信度筛选后框数量: {}".format(len(pred_filtered_by_conf)))
    if len(pred_filtered_by_conf) == 0:
        return []

    # 2. 计算原始坐标（640x640图）
    x_center = pred_filtered_by_conf[:, 0]
    y_center = pred_filtered_by_conf[:, 1]
    w = pred_filtered_by_conf[:, 2]
    h = pred_filtered_by_conf[:, 3]
    if len(pred_filtered_by_conf) > 0:
        print("  模型输出xywh（640x640图）: x={:.0f}, y={:.0f}, w={:.0f}, h={:.0f}".format(
            x_center[0], y_center[0], w[0], h[0]
        ))

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    if len(pred_filtered_by_conf) > 0:
        print("  转换后x1y1x2y2（640x640图）: {:.0f}, {:.0f}, {:.0f}, {:.0f}".format(
            x1[0], y1[0], x2[0], y2[0]
        ))

    # 3. 还原到原始图像坐标（减去填充 + 缩放）
    x1 -= pad_w
    y1 -= pad_h
    x2 -= pad_w
    y2 -= pad_h
    if len(pred_filtered_by_conf) > 0:
        print("  减去填充后坐标（640x640图）: {:.0f}, {:.0f}, {:.0f}, {:.0f}".format(
            x1[0], y1[0], x2[0], y2[0]
        ))

    x1 /= r
    y1 /= r
    x2 /= r
    y2 /= r
    if len(pred_filtered_by_conf) > 0:
        print("  还原到原始图坐标: {:.0f}, {:.0f}, {:.0f}, {:.0f}".format(
            x1[0], y1[0], x2[0], y2[0]
        ))

    # 4. 生成boxes并限制坐标范围
    boxes = np.stack((x1, y1, x2, y2), axis=1)
    boxes = np.clip(boxes, 0, [w0, h0, w0, h0])
    if len(pred_filtered_by_conf) > 0:
        print("  限制后最终坐标: {:.0f}, {:.0f}, {:.0f}, {:.0f}".format(
            boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
        ))

    # 5. 检测区域过滤：只保留图像上N%区域内的框（核心修改）
    area_threshold = h0 * (DETECTION_AREA_TOP_PERCENT / 100)  # 计算区域阈值（像素）
    area_mask = boxes[:, 3] <= area_threshold  # 筛选y2（框右下角）≤ 阈值的框
    boxes_filtered_by_area = boxes[area_mask]
    confs_filtered_by_area = pred_filtered_by_conf[:, 4][area_mask]  # 对应置信度
    clss_filtered_by_area = np.zeros_like(confs_filtered_by_area, dtype=int)  # 对应类别
    print(f"  检测区域筛选后框数量: {len(boxes_filtered_by_area)} (只保留上{DETECTION_AREA_TOP_PERCENT}%区域)")
    if len(boxes_filtered_by_area) == 0:
        return []

    # 6. NMS非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes_filtered_by_area.tolist(), confs_filtered_by_area.tolist(), conf_thres, iou_thres)
    if isinstance(indices, (int, np.integer)):  
        indices = np.array([indices])
    nms_count = len(indices) if isinstance(indices, np.ndarray) else 0
    print("  NMS后保留框数量: {}".format(nms_count))

    # 7. 整理最终结果
    results = []
    if isinstance(indices, np.ndarray):
        for i in indices.flatten():
            if i < len(boxes_filtered_by_area):  
                x1, y1, x2, y2 = map(int, boxes_filtered_by_area[i])
                results.append((x1, y1, x2, y2, round(confs_filtered_by_area[i], 4), clss_filtered_by_area[i]))
    return results


def read_prev_count(txt_path):
    """读取历史计数（不存在则返回0）"""
    try:
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                match = re.search(r'(\d+)辆', content)
                if match:
                    return int(match.group(1))
        return 0
    except Exception as e:
        print("读取历史计数失败，初始化为0: {}".format(e))
        return 0


def save_count(txt_path, total_count):
    """保存累加后的计数"""
    try:
        content = "A街区有{}辆电动车违停".format(total_count)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  计数已保存: {} → {}".format(txt_path, content))
        return True
    except Exception as e:
        print("保存计数失败: {}".format(e))
        return False


# -------------------------- 主函数 --------------------------
def main():
    # 初始化ROS节点
    rospy.init_node("yolo_ddc_detection", anonymous=True)
    
    # 初始化保存目录和文件名
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)  
    count_txt_path = os.path.join(TXT_OUTPUT_DIR, COUNT_TXT_NAME)  
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    detected_img_name = "detected_{}.jpg".format(timestamp)
    detected_img_path = os.path.join(OUTPUT_DIR, detected_img_name)

    print("=== Output Directory: {} ===".format(OUTPUT_DIR))
    print("=== TXT Output Directory: {} ===".format(TXT_OUTPUT_DIR))  
    print("  重要：请打开 {} 查看图片，{} 查看计数txt！\n".format(OUTPUT_DIR, TXT_OUTPUT_DIR))

    # 加载ONNX模型
    try:
        det_session = InferenceSession(
            DET_MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        det_input_name = det_session.get_inputs()[0].name
        det_output_names = [out.name for out in det_session.get_outputs()]
        det_input_shape = det_session.get_inputs()[0].shape
        print("Detection Model Input Shape: {} (batch, channel, height, width)".format(det_input_shape))
        print("Successfully loaded detection model: {}\n".format(DET_MODEL_PATH))
    except Exception as e:
        print("Failed to load detection model: {}".format(e))
        return

    print("=" * 60)
    print("ROS摄像头话题识别 + ddc目标检测 + 累加计数保存（对齐ROS参数）")
    print(f"检测区域：图像上{DETECTION_AREA_TOP_PERCENT}%区域（可在配置区修改）")
    print("=" * 60)
    print(" 摄像头话题: {}".format(CAMERA_TOPIC))
    print(" 读取帧数: {}".format(CAPTURE_NUM))
    print(" 保存目录: {}".format(OUTPUT_DIR))
    print(" 检测模型路径: {}".format(DET_MODEL_PATH))
    print("  检测标签: 0 → 'ddc'（电动车）")
    print(" 保存文件: {}（标注图，时间戳命名）、{}（累加计数）".format(
        detected_img_name, COUNT_TXT_NAME
    ))
    print("=" * 60)

    # 1. 从ROS话题读取摄像头帧（替代原本地摄像头读取）
    frame = get_ros_camera_frame(CAMERA_TOPIC, CAMERA_TIMEOUT)
    if frame is None:
        rospy.logerr("获取ROS摄像头帧失败，程序退出")
        return

    # 2. 预处理
    img_input, r, pad = preprocess_image(frame)
    if img_input is None:
        print(" 图像预处理失败")
        return
    print("  缩放比例: {:.4f}, 填充量: (pad_w={}, pad_h={})".format(r, pad[0], pad[1]))

    # 3. 目标检测推理
    preds = det_session.run(det_output_names, {det_input_name: img_input})
    pred_boxes = preds[0]
    print("  模型输出形状: {}".format(pred_boxes.shape))

    # 4. 后处理
    detection_results = postprocess_output(pred_boxes, frame.shape[:2], r, pad, CONF_THRESHOLD, IOU_THRESHOLD)
    current_count = len(detection_results)  
    print("\n 本次检测到 {} 辆电动车".format(current_count))

    # 5. 绘制检测框
    for (x1, y1, x2, y2, conf, cls) in detection_results:
        label = LABEL_MAP.get(cls, "cls_{}".format(cls))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        label_text = "{} {:.2f}".format(label, conf)
        cv2.putText(frame, label_text, (x1, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ========== 新增：绘制识别区域分割线 ==========
    h0, w0 = frame.shape[:2]  # 获取原始图像尺寸
    divide_y = int(h0 * (DETECTION_AREA_TOP_PERCENT / 100))  # 计算分割线y坐标
    # 绘制红色分割线（BGR: 0,0,255），线宽2
    cv2.line(frame, (0, divide_y), (w0, divide_y), (0, 0, 255), 2)
    # 绘制分割线说明文字（红色）
    cv2.putText(frame, f"Detection Area (Top {DETECTION_AREA_TOP_PERCENT}%)", 
                (10, divide_y - 10),  # 文字位置（分割线上方10像素）
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # =============================================

    # 6. 保存标注后的图片
    cv2.imwrite(detected_img_path, frame)
    print("  标注图片已保存: {}".format(detected_img_path))

    # 7. 累加计数并保存
    prev_count = read_prev_count(count_txt_path)
    total_count = prev_count + current_count
    save_count(count_txt_path, total_count)

    # 最终结果打印
    print("\n" + "=" * 60)
    print(" 处理完成！")
    print(" 历史计数: {} 辆".format(prev_count))
    print(" 本次新增: {} 辆".format(current_count))
    print(" 累计总数: {} 辆".format(total_count))
    print(" 结果文件：")
    print("   - 标注图: {}（时间戳命名，不覆盖）".format(detected_img_path))
    print("   - 计数文件: {} (内容：A街区有{}辆电动车违停)".format(
        count_txt_path, total_count
    ))
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  收到中断信号，程序退出")
    except Exception as e:
        print("\n\n 程序异常退出: {}".format(str(e)))
    finally:
        # 确保ROS节点正常退出
        if rospy.core.is_initialized():
            rospy.signal_shutdown("程序正常退出")
