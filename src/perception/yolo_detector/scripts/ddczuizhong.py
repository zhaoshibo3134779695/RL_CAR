#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS摄像头目标检测 - 增强版物体跟踪与计数系统
修复物体重复计数问题，增加方向距离计算
【核心检测逻辑已对齐onnx_yolo_detector_node.py】
【已删除图片中spatial relations的显示】
【新增：仅目标通过屏幕右侧1/3竖线时才计数】
【新增：结果保存为ddc.txt】
【新增：只识别屏幕下半部分，绘制识别区域分割线】
"""
import rospy
import cv2
import numpy as np
import os
import time
import math
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from onnxruntime import InferenceSession
from runtime_paths import EBIKE_MODEL, RUNTIME_DDC_IMG, RUNTIME_DDC_TXT, ensure_runtime_dirs

# ========== 核心配置区域 ==========
# ROS摄像头配置
CAMERA_TOPIC = '/camera/color/image_raw'  # 相机话题
DURATION = 30  # 持续检测时间（秒）
FRAME_RATE = 10  # 目标帧率（每秒帧数）

# 目标检测配置【对齐launch参数】
DET_MODEL_PATH = str(EBIKE_MODEL)
LABEL_MAP = {0: "dao", 1: "zc"}  # 类别映射：0=倒伏(dao)，1=正常(zc)
INPUT_SIZE = 640  # 模型输入尺寸（统一为int，兼容letterbox）
CONF_THRESHOLD = 0.3  # 置信度阈值
IOU_THRESHOLD = 0.1  # 非极大值抑制阈值
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # dao-红, zc-绿 (BGR)

# 跟踪参数 - 增强配置
TRACKING_DISTANCE_THRESHOLD = 100  # 像素距离阈值，判断是否为同一物体
MAX_TRACKING_FRAMES = 30  # 最大跟踪帧数（物体消失后仍保留的帧数）
DIRECTION_CONSISTENCY_THRESHOLD = 0.7  # 方向一致性阈值（余弦相似度）
POSITION_VARIATION_THRESHOLD = 50  # 位置变化阈值（像素）

# 计数线配置（右侧1/3处）
COUNT_LINE_RATIO = 2/3  # 计数线位置：图像宽度的2/3处（右侧1/3起始点）
COUNT_LINE_COLOR = (255, 0, 255)  # 计数线颜色（洋红色）
COUNT_LINE_THICKNESS = 2  # 计数线粗细

# 下半部分识别配置【新增】
HALF_HEIGHT_RATIO = 0.5  # 只识别图像下半部分（从50%高度开始，可调整）
HALF_AREA_LINE_COLOR = (255, 255, 0)  # 下半部分分割线颜色（黄色）
HALF_AREA_LINE_THICKNESS = 2  # 分割线粗细
HALF_AREA_LABEL = "Detection Area (Lower Half)"  # 识别区域标签
HALF_AREA_ALPHA = 0.1  # 下半部分识别区域半透明填充透明度

# 保存配置
SAVE_DIR = str(RUNTIME_DDC_IMG)
DDC_TXT_PATH = str(RUNTIME_DDC_TXT / "ddczuizhong.txt")
# ==================================

# 全局变量初始化
bridge = CvBridge()
start_time = None
frame_count = 0  # 已处理帧数
is_running = True  # 是否继续检测
ensure_runtime_dirs()

# -------------------------- 【修改】结果保存函数 --------------------------
def save_ddc_result(zc_count, dao_count):
    """保存最终计数结果到ddczuizhong.txt"""
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    txt_path = DDC_TXT_PATH
    
    # 构造保存内容
    result_text = f"停车区内电动车正常{zc_count}辆，倒伏{dao_count}辆"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result_text)
    
    print(f"\n ✓ 最终计数结果已保存到: {txt_path}")
    print(f"   保存内容: {result_text}")
    return txt_path, result_text

def broadcast_ddc_result(zc_count, dao_count):
    """语音播报最终计数结果（非阻塞版）"""
    try:
        # 构造播报文本
        broadcast_text = f"停车区内电动车正常{zc_count}辆，倒伏{dao_count}辆"
        print(f"\n 开始非阻塞语音播报: {broadcast_text}")
        
    except Exception as e:
        print(f" 语音播报初始化失败: {e}")

# -------------------------- 【新增】对齐官方节点的核心工具函数 --------------------------
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """YOLOv8标准letterbox预处理（替换原preprocess_image）"""
    shape = img.shape[:2]  # current shape: (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # Compute padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # Resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Padding
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color)

    # Ensure exact shape
    img = img[:new_shape[0], :new_shape[1]]

    return img, r, (dw, dh)

def standardize_output(output):
    """统一YOLOv8 ONNX输出格式（替换原硬编码transpose）"""
    arr = output[0]
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[1] < arr.shape[2]:     # (1, 8400, nc+4)
            return arr.transpose(1, 0)[0]
        else:
            return arr.transpose(2, 1, 0)[:, :, 0]

    elif arr.ndim == 2:
        return arr if arr.shape[0] > arr.shape[1] else arr.T

    raise RuntimeError("Unsupported output shape: {}".format(arr.shape))

def nms_optimized(boxes, scores, iou_thres=0.45):
    """官方节点的NMS实现（替换原手写NMS）"""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(xx2 - xx1, 0)
        h = np.maximum(yy2 - yy1, 0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        keep_idx = np.where(iou <= iou_thres)[0]
        order = order[keep_idx + 1]

    return keep

# -------------------------- 增强版物体跟踪器类（修改计数逻辑） --------------------------
class EnhancedObjectTracker:
    def __init__(self, distance_threshold=100, max_frames=30, 
                 direction_threshold=0.7, position_variation=50):
        self.distance_threshold = distance_threshold
        self.max_frames = max_frames
        self.direction_threshold = direction_threshold
        self.position_variation = position_variation
        
        # 存储跟踪的对象信息
        self.objects = {}  # {id: (center, cls_id, frames_since_seen, trajectory, last_direction, appearance_count)}
        self.next_id = 0
        self.class_counts = {0: 0, 1: 0}  # 各类别计数：0=dao(倒伏), 1=zc(正常)
        self.counted_by_line = set()  # 新增：已通过右侧1/3计数线的物体ID
        self.line_x = None  # 新增：计数线的x坐标
        
        # 方向映射
        self.direction_map = {
            "left": (-1, 0), "right": (1, 0), 
            "up": (0, -1), "down": (0, 1),
            "left_up": (-1, -1), "right_up": (1, -1),
            "left_down": (-1, 1), "right_down": (1, 1)
        }

    def calculate_center(self, bbox):
        """计算边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_direction_vector(self, point1, point2):
        """计算从point1到point2的方向向量"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        distance = math.sqrt(dx**2 + dy**2)
        if distance == 0:
            return (0, 0), 0
        return (dx/distance, dy/distance), distance

    def direction_similarity(self, vec1, vec2):
        """计算两个方向向量的余弦相似度"""
        if vec1 == (0, 0) or vec2 == (0, 0):
            return 0
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        return dot_product  # 因为都是单位向量，所以模长为1

    def get_direction_name(self, vector):
        """根据方向向量获取方向名称"""
        if vector == (0, 0):
            return "static"
        
        dx, dy = vector
        angle = math.atan2(dy, dx) * 180 / math.pi
        
        if -22.5 <= angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "right_down"
        elif 67.5 <= angle < 112.5:
            return "down"
        elif 112.5 <= angle < 157.5:
            return "left_down"
        elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
            return "left"
        elif -157.5 <= angle < -112.5:
            return "left_up"
        elif -112.5 <= angle < -67.5:
            return "up"
        else:  # -67.5 <= angle < -22.5
            return "right_up"

    def update(self, detections, img_width):
        """增强版更新跟踪器状态【新增img_width参数，修改计数逻辑】"""
        # 计算右侧1/3计数线位置
        self.line_x = img_width * COUNT_LINE_RATIO
        current_centers = {}
        used_object_ids = set()

        # 第一阶段：为当前检测到的物体寻找最佳匹配
        for detection in detections:
            x1, y1, x2, y2, conf, cls_id = detection
            current_center = self.calculate_center((x1, y1, x2, y2))
            current_centers[current_center] = (detection, cls_id)

            best_match_id = None
            best_match_score = -1
            current_direction_vector, current_distance = self.calculate_direction_vector((0, 0), (0, 0))

            # 对每个已跟踪的物体计算匹配分数
            for obj_id, obj_data in self.objects.items():
                if obj_id in used_object_ids:
                    continue
                    
                last_center, last_cls_id, frames_since_seen, trajectory, last_direction, appearance_count = obj_data
                
                # 类别必须相同
                if cls_id != last_cls_id:
                    continue
                
                # 计算距离
                distance = self.calculate_distance(current_center, last_center)
                
                # 计算方向一致性（如果存在历史轨迹）
                direction_similarity = 0
                if len(trajectory) >= 2:
                    prev_center = trajectory[-2]  # 上一个位置
                    history_direction, _ = self.calculate_direction_vector(prev_center, last_center)
                    current_direction, current_distance = self.calculate_direction_vector(last_center, current_center)
                    direction_similarity = self.direction_similarity(history_direction, current_direction)
                
                # 综合评分（距离越近、方向越一致分数越高）
                distance_score = max(0, 1 - distance / self.distance_threshold)
                direction_score = max(0, direction_similarity)
                
                # 最终评分（距离权重0.7，方向权重0.3）
                total_score = 0.7 * distance_score + 0.3 * direction_score
                
                if total_score > best_match_score and distance < self.distance_threshold:
                    best_match_score = total_score
                    best_match_id = obj_id
                    current_direction_vector, _ = self.calculate_direction_vector(last_center, current_center)

            if best_match_id is not None and best_match_score > 0.3:
                # 更新已存在的物体
                obj_data = self.objects[best_match_id]
                last_center, last_cls_id, frames_since_seen, trajectory, last_direction, appearance_count = obj_data
                
                # 更新轨迹和方向
                trajectory.append(current_center)
                if len(trajectory) > 10:  # 保持最近10个位置
                    trajectory = trajectory[-10:]
                
                self.objects[best_match_id] = (
                    current_center, cls_id, 0, trajectory, 
                    current_direction_vector, appearance_count + 1
                )
                used_object_ids.add(best_match_id)
                current_centers[current_center] = None  # 标记为已匹配
                
                # 记录方向信息
                direction_name = self.get_direction_name(current_direction_vector)
                rospy.loginfo(f"物体 {best_match_id} 移动方向: {direction_name}, 距离: {self.calculate_distance(last_center, current_center):.1f}像素")

                # 【新增】检查是否越过右侧1/3计数线且未计数
                if best_match_id not in self.counted_by_line and current_center[0] >= self.line_x:
                    self.class_counts[cls_id] += 1
                    self.counted_by_line.add(best_match_id)
                    rospy.loginfo(f"物体 {best_match_id} ({LABEL_MAP[cls_id]}) 越过计数线，计数+1 | 计数线位置: {self.line_x:.1f}px")

            else:
                # 创建新物体（仅跟踪，不立即计数）
                obj_id = self.next_id
                trajectory = [current_center]
                self.objects[obj_id] = (current_center, cls_id, 0, trajectory, (0, 0), 1)
                self.next_id += 1
                used_object_ids.add(obj_id)
                
                # 【新增】检查新物体是否初始位置就在计数线右侧
                if obj_id not in self.counted_by_line and current_center[0] >= self.line_x:
                    self.class_counts[cls_id] += 1
                    self.counted_by_line.add(obj_id)
                    rospy.loginfo(f"新物体 {obj_id} ({LABEL_MAP[cls_id]}) 初始位置在计数线右侧，计数+1 | 计数线位置: {self.line_x:.1f}px")
                else:
                    rospy.loginfo(f"新物体 {obj_id} ({LABEL_MAP[cls_id]}) 被跟踪（未过计数线）| 计数线位置: {self.line_x:.1f}px")

        # 第二阶段：更新未匹配的物体（增加未出现帧数）
        objects_to_remove = []
        for obj_id, obj_data in self.objects.items():
            center, cls_id, frames_since_seen, trajectory, last_direction, appearance_count = obj_data
            
            if obj_id not in used_object_ids:
                frames_since_seen += 1
                # 保持其他信息不变，只更新未出现帧数
                self.objects[obj_id] = (center, cls_id, frames_since_seen, trajectory, last_direction, appearance_count)

                # 如果物体消失时间过长，移除
                if frames_since_seen > self.max_frames:
                    objects_to_remove.append(obj_id)
                    rospy.loginfo(f"物体 {obj_id} 因长时间消失被移除")

        # 移除长时间未出现的物体
        for obj_id in objects_to_remove:
            if obj_id in self.objects:
                # 移除时如果已计数，不影响最终计数
                del self.objects[obj_id]

    def get_counts(self):
        """获取当前计数"""
        return self.class_counts.copy()

    def get_object_count(self):
        """获取当前跟踪的物体数量"""
        return len(self.objects)
    
    def get_object_positions(self):
        """获取所有物体的位置和ID信息，用于判断左右关系"""
        positions = {}
        for obj_id, obj_data in self.objects.items():
            center, cls_id, frames_since_seen, trajectory, last_direction, appearance_count = obj_data
            positions[obj_id] = {
                'center': center,
                'class': cls_id,
                'class_name': LABEL_MAP.get(cls_id, 'unknown'),
                'frames_missing': frames_since_seen,
                'is_counted': obj_id in self.counted_by_line,  # 新增：标记是否已计数
                'line_x': self.line_x  # 新增：返回计数线位置
            }
        return positions

# 初始化增强版物体跟踪器（完全保留）
tracker = EnhancedObjectTracker(
    distance_threshold=TRACKING_DISTANCE_THRESHOLD,
    max_frames=MAX_TRACKING_FRAMES,
    direction_threshold=DIRECTION_CONSISTENCY_THRESHOLD,
    position_variation=POSITION_VARIATION_THRESHOLD
)

# -------------------------- 初始化设置（完全保留，新增语音配置提示） --------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"=== ROS增强版目标检测系统启动 ===")
print(f" 摄像头话题: {CAMERA_TOPIC}")
print(f" 检测模型路径: {DET_MODEL_PATH}")
print(f" 持续检测时间: {DURATION}秒 | 目标帧率: {FRAME_RATE}帧/秒")
print(f" 跟踪参数: 距离阈值={TRACKING_DISTANCE_THRESHOLD}px")
print(f" 增强功能: 方向感知、轨迹跟踪、防重复计数、右侧1/3计数线")
print(f" 计数规则: 仅目标通过图像{COUNT_LINE_RATIO*100:.0f}%宽度处竖线才计数")
print(f" 识别范围: 仅图像下半部分（{HALF_HEIGHT_RATIO*100:.0f}%高度以下）")  # 新增提示
print(f" 结果保存目录: {SAVE_DIR}\n")

# 加载目标检测ONNX模型【优化：兼容CUDA】
try:
    # 兼容CUDA/CPU
    providers = ["CPUExecutionProvider"]
    if rospy.get_param("~use_cuda", False):  # 可从launch读取use_cuda参数
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    det_session = InferenceSession(
        DET_MODEL_PATH,
        providers=providers
    )
    det_input_name = det_session.get_inputs()[0].name
    det_input_shape = det_session.get_inputs()[0].shape
    print(f" 检测模型加载成功 - 输入形状: {det_input_shape} | 推理设备: {providers[0]}")
    print(f"====================================\n")
except Exception as e:
    print(f" 检测模型加载失败: {e}")
    exit(1)

# -------------------------- 【修改】核心检测流程（对齐官方节点） --------------------------
def process_detection(frame):
    """
    替换原preprocess_image + postprocess_output
    核心检测流程：letterbox预处理 → 模型推理 → 标准化输出 → 后处理 → NMS
    """
    h0, w0 = frame.shape[:2]
    
    # 1. 标准letterbox预处理（BGR→RGB + 等比例缩放 + 归一化）
    img_lb, r, (dw, dh) = letterbox(frame, INPUT_SIZE)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)  # 关键：BGR转RGB
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None]  # HWC→CHW + batch维度
    
    # 2. 模型推理
    preds = det_session.run(None, {det_input_name: img_input})
    
    # 3. 标准化输出（兼容所有YOLOv8 ONNX格式）
    preds_standard = standardize_output(preds)
    
    # 4. 置信度过滤
    num_classes = preds_standard.shape[1] - 4
    boxes = preds_standard[:, :4]  # xywh
    scores = preds_standard[:, 4:]
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]
    mask = confs >= CONF_THRESHOLD
    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # 5. xywh → xyxy + 还原到原始图像尺寸（关键修正）
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2 - dw  # 减去填充偏移
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2 - dh
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2 - dw
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2 - dh
    xyxy /= r  # 还原缩放比例
    
    # 6. 边界裁剪（防止超出图像范围）
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w0)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h0)
    
    # 7. 优化版NMS
    keep = nms_optimized(xyxy, confs, IOU_THRESHOLD)
    xyxy, confs, cls_ids = xyxy[keep], confs[keep], cls_ids[keep]
    
    # 8. 构造检测结果（与原格式一致）
    detections = []
    for (x1, y1, x2, y2), sc, cid in zip(xyxy.astype(int), confs, cls_ids):
        detections.append((x1, y1, x2, y2, round(sc, 4), int(cid)))
    
    return detections

# -------------------------- 其余函数（新增计数线绘制+下半部分识别区域绘制） --------------------------
def analyze_spatial_relationships(positions):
    """分析物体间的空间关系（左右位置）"""
    if len(positions) < 2:
        return []
    
    relationships = []
    obj_ids = list(positions.keys())
    
    # 比较每对物体的位置
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            id1, id2 = obj_ids[i], obj_ids[j]
            pos1 = positions[id1]['center']
            pos2 = positions[id2]['center']
            
            # 判断左右关系
            if pos1[0] < pos2[0]:
                left_id, right_id = id1, id2
                left_pos, right_pos = pos1, pos2
            else:
                left_id, right_id = id2, id1
                left_pos, right_pos = pos2, pos1
            
            distance = math.sqrt((left_pos[0] - right_pos[0])**2 + (left_pos[1] - right_pos[1])**2)
            relationships.append({
                'left': left_id,
                'right': right_id,
                'distance': distance,
                'left_class': positions[left_id]['class_name'],
                'right_class': positions[right_id]['class_name']
            })
    
    return relationships

def draw_enhanced_detection_boxes(img, detection_results, current_counts, tracked_objects_count, spatial_relations):
    """绘制增强版检测框和空间关系信息【新增计数线+下半部分识别区域绘制】"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]

    # 【新增】绘制下半部分识别区域分割线（黄色水平线）
    half_y = int(h * HALF_HEIGHT_RATIO)
    cv2.line(img_copy, (0, half_y), (w, half_y), HALF_AREA_LINE_COLOR, HALF_AREA_LINE_THICKNESS)
    # 绘制识别区域标签
    cv2.putText(img_copy, HALF_AREA_LABEL, (10, half_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, HALF_AREA_LINE_COLOR, 2)
    # 半透明填充下半部分识别区域（可选，增强视觉效果）
    overlay = img_copy.copy()
    cv2.rectangle(overlay, (0, half_y), (w, h), HALF_AREA_LINE_COLOR, -1)
    cv2.addWeighted(overlay, HALF_AREA_ALPHA, img_copy, 1 - HALF_AREA_ALPHA, 0, img_copy)

    # 绘制右侧1/3计数竖线
    line_x = int(w * COUNT_LINE_RATIO)
    cv2.line(img_copy, (line_x, 0), (line_x, h), COUNT_LINE_COLOR, COUNT_LINE_THICKNESS)
    cv2.putText(img_copy, "Count Line", (line_x - 60, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COUNT_LINE_COLOR, 2)

    # 绘制检测结果（完全保留）
    for (x1, y1, x2, y2, conf, cls_id) in detection_results:
        cls_name = LABEL_MAP.get(cls_id, f"未知类别{cls_id}")
        label = f"{cls_name} {conf:.2f}"
        color = CLASS_COLORS.get(cls_id, (0, 255, 255))

        # 绘制边界框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # 绘制中心点和ID
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img_copy, (center_x, center_y), 4, color, -1)
        
        # 获取物体ID和计数状态
        obj_positions = tracker.get_object_positions()
        for obj_id, pos_info in obj_positions.items():
            if abs(pos_info['center'][0] - center_x) < 10 and abs(pos_info['center'][1] - center_y) < 10:
                # 标记是否已计数
                count_label = "Counted" if pos_info['is_counted'] else "Uncounted"
                cv2.putText(img_copy, f"ID:{obj_id} ({count_label})", (x1, y1 - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                break

        # 绘制标签
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
        cv2.rectangle(img_copy, (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0] + 5, label_y + 5), color, -1)
        cv2.putText(img_copy, label, (x1 + 2, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 添加增强版统计信息面板
    info_lines = [
        f"Frame: {frame_count}",
        f"Tracked Objects: {tracked_objects_count}",
        f"DAO Count: {current_counts[0]}",
        f"ZC Count: {current_counts[1]}",
        f"Count Line: X = {line_x}px",
        f"Detection Area: Y ≥ {half_y}px (Lower Half)"  # 新增：显示下半部分识别区域
    ]

    # 绘制信息面板
    panel_height = len(info_lines) * 25 + 20
    cv2.rectangle(img_copy, (10, 10), (500, panel_height), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (10, 10), (500, panel_height), (255, 255, 255), 2)

    # 绘制文本信息
    for i, line in enumerate(info_lines):
        y_position = 35 + i * 25
        cv2.putText(img_copy, line, (20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_copy

def save_detection_image(frame_idx, img, detection_results, current_counts, tracked_objects_count, spatial_relations):
    """保存带检测框和跟踪信息的图片"""
    global SAVE_DIR

    # 绘制检测框和跟踪信息
    img_with_boxes = draw_enhanced_detection_boxes(
        img, detection_results, current_counts, tracked_objects_count, spatial_relations
    )

    # 保存图片
    img_path = os.path.join(SAVE_DIR, f"frame_{frame_idx:03d}.jpg")
    cv2.imwrite(img_path, img_with_boxes)

    print(f" 第{frame_idx:03d}帧图片已保存: {img_path}")

# -------------------------- ROS回调函数（传入图像宽度到tracker + 下半部分过滤） --------------------------
def image_callback(msg):
    global frame_count, is_running, start_time, tracker

    # 停止条件
    if not is_running or (time.time() - start_time) > DURATION:
        is_running = False
        return

    try:
        # 1. 从ROS消息转换为OpenCV图像
        cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        h0, w0 = cv_img.shape[:2]
        frame_count += 1
        elapsed_time = time.time() - start_time
        print(f"\n[帧{frame_count:03d} | 耗时{elapsed_time:.2f}秒] 收到图像 - 原始尺寸: {h0}x{w0}")

        # 2. 调用对齐官方节点的检测流程
        detection_results = process_detection(cv_img)

        # 【新增】过滤只保留下半部分的检测结果
        half_y = int(h0 * HALF_HEIGHT_RATIO)
        original_detections = detection_results.copy()
        filtered_detections = []
        for det in original_detections:
            x1, y1, x2, y2, conf, cls_id = det
            # 计算检测框中心点的y坐标（也可改为判断框的底部y2 >= half_y）
            center_y = (y1 + y2) // 2
            # 只保留中心点在下半部分的检测结果
            if center_y >= half_y:
                filtered_detections.append(det)
        detection_results = filtered_detections
        print(f" 下半部分过滤: 原始{len(original_detections)}个目标 → 过滤后{len(detection_results)}个（仅保留y≥{half_y}px的目标）")

        # 3. 更新增强版物体跟踪器【新增传入图像宽度w0】
        tracker.update(detection_results, w0)
        current_counts = tracker.get_counts()
        tracked_objects_count = tracker.get_object_count()
        
        # 4. 分析空间关系
        obj_positions = tracker.get_object_positions()
        spatial_relations = analyze_spatial_relationships(obj_positions)

        # 5. 打印增强版检测和跟踪结果
        if detection_results:
            print(f" 检测到 {len(detection_results)} 个目标（下半部分）：")
            for idx, (x1, y1, x2, y2, conf, cls_id) in enumerate(detection_results, 1):
                cls_name = LABEL_MAP.get(cls_id, f"未知类别{cls_id}")
                # 打印目标是否过计数线
                center_x = (x1 + x2) // 2
                is_over_line = center_x >= (w0 * COUNT_LINE_RATIO)
                line_status = " 过计数线" if is_over_line else " 未过计数线"
                print(f"  目标{idx}: {cls_name} | 置信度: {conf:.4f} | 坐标: ({x1},{y1})-({x2},{y2}) | {line_status}")
        else:
            print(f" 下半部分未检测到任何目标（置信度阈值: {CONF_THRESHOLD}）")
        
        print(f" 跟踪统计: 跟踪物体数={tracked_objects_count}, DAO计数={current_counts[0]}, ZC计数={current_counts[1]}")
        print(f" 计数线位置: X = {w0 * COUNT_LINE_RATIO:.1f}px (图像宽度{w0}px的{COUNT_LINE_RATIO*100:.0f}%)")
        print(f" 识别区域: Y ≥ {half_y}px (图像高度{h0}px的{HALF_HEIGHT_RATIO*100:.0f}%)")
        
        # 打印空间关系
        if spatial_relations:
            print(" 空间关系分析:")
            for rel in spatial_relations:
                print(f"  ID{rel['left']}({rel['left_class']}) 在 ID{rel['right']}({rel['right_class']}) 左边, 距离: {rel['distance']:.1f}像素")

        # 6. 保存带增强信息的检测图片
        save_detection_image(frame_count, cv_img, detection_results, current_counts, tracked_objects_count, spatial_relations)

    except Exception as e:
        print(f" 第{frame_count}帧处理失败: {str(e)}")

# -------------------------- 主函数 --------------------------
def main():
    global start_time, is_running, tracker

    # 初始化ROS节点
    rospy.init_node('enhanced_ros_onnx_object_detection', anonymous=True)

    # 订阅摄像头话题
    sub = rospy.Subscriber(CAMERA_TOPIC, Image, image_callback, queue_size=FRAME_RATE * 2)

    # 记录开始时间
    start_time = time.time()
    print(f" 开始增强版检测（持续{DURATION}秒）...")

    # 控制帧率并等待检测结束
    rate = rospy.Rate(FRAME_RATE)
    while not rospy.is_shutdown() and is_running:
        if (time.time() - start_time) > DURATION:
            is_running = False
            break
        rate.sleep()

    # 取消订阅
    sub.unregister()

    # 打印最终统计
    elapsed_total = time.time() - start_time
    final_counts = tracker.get_counts()
    final_positions = tracker.get_object_positions()
    
    print("\n" + "=" * 60)
    print(f" 增强版检测完成！")
    print(f" 统计信息：")
    print(f"  - 总耗时: {elapsed_total:.2f}秒")
    print(f"  - 实际处理帧数: {frame_count}帧")
    print(f"  - 实际帧率: {frame_count / elapsed_total:.2f}帧/秒")
    print(f"  - 最终计数 - DAO(倒伏): {final_counts[0]}, ZC(正常): {final_counts[1]}")
    print(f"  - 最大物体ID: {tracker.next_id - 1}")
    print(f"  - 图片保存目录: {SAVE_DIR}")
    print(f"  - 计数规则: 仅目标通过图像{COUNT_LINE_RATIO*100:.0f}%宽度处竖线才计数")
    print(f"  - 识别范围: 仅图像下半部分（{HALF_HEIGHT_RATIO*100:.0f}%高度以下）")  # 新增
    
    # 打印最终的物体位置关系
    if final_positions:
        print("最终物体位置:")
        for obj_id, pos_info in final_positions.items():
            count_status = "已计数" if pos_info['is_counted'] else "未计数"
            print(f"  物体{obj_id} ({pos_info['class_name']}): 位置{pos_info['center']}, 消失帧数: {pos_info['frames_missing']}, 状态: {count_status}")
    print("=" * 60)

    # 保存结果到ddc.txt
    zc_count = final_counts[1]  # 正常车辆数
    dao_count = final_counts[0] # 倒伏车辆数
    save_ddc_result(zc_count, dao_count)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n 收到中断信号，程序退出")
    except Exception as e:
        print(f"\n 程序异常退出: {str(e)}")
