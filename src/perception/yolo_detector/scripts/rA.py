#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS摄像头实时识别 + YOLO ONNX（renwu.onnx） + 结果输出(0/1) + 图片保存 + 众数统计保存TXT
适配：renwu.onnx共3类，仅使用前2类（class_0/class_1），过滤第3类
修改：每次识别保留1张匹配众数的图片（历史图片不删除）
修改2：TXT文件改为2.txt，累计叠加X/Y并按指定格式写入
新增：识别框面积阈值过滤，仅面积大于阈值的框才计数和显示
新增2：不同类别用不同颜色框标注 + 仅识别摄像头下半区域
"""
import rospy
import cv2
import numpy as np
import os
import time
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from onnxruntime import InferenceSession
from runtime_paths import PEOPLE_MODEL, RUNTIME_R, ensure_runtime_dirs


# ========== 核心配置区域（适配3类模型，仅用前2类） ==========
# ROS摄像头配置
CAMERA_TOPIC = '/camera/color/image_raw'  # 相机话题
CAPTURE_NUM = 3                          # 捕获帧数（避免1帧无目标）
DETECT_REGION_RATIO = 0.5                # 仅识别下半区域（0.5表示只处理图片的后50%区域）

# 模型与保存配置
MODEL_PATH = str(PEOPLE_MODEL)
SAVE_DIR = str(RUNTIME_R)
IMG_SIZE = 640                              # 模型输入尺寸
CONF_THRESH = 0.1                          # 置信度阈值（可适当调低至0.1）
IOU_THRESH = 0.3                           # NMS阈值
CLASS_NAMES = ["class_0", "class_1"]        # 仅使用前2类（对应输出0/1）
IGNORE_CLASS_ID = 2                         # 忽略第3类（索引2）
TXT_FILE_NAME = "rA.txt"                     # 新增：指定TXT文件名为2.txt
BBOX_AREA_THRESHOLD = 5000                  # 识别框面积阈值（像素），大于此值才计数和显示
# 不同类别对应的框颜色（BGR格式）
CLASS_COLORS = {
    0: (255, 0, 0),    # class_0: 蓝色
    1: (0, 0, 255)     # class_1: 红色
}

# 全局变量
bridge = CvBridge()
count = 0  # 已处理帧数
saved_files = []  # 保存的文件列表（本次运行）
frame_counts = []  # 存储每帧的(class0_num, class1_num)
current_run_imgs = []  # 新增：记录本次运行生成的所有图片路径（仅本次）
ensure_runtime_dirs()
# ==================================


# ============================================================
# 众数计算函数（兼容空列表、多众数场景）
# ============================================================
def get_mode(arr):
    """
    计算列表的众数（出现次数最多的值）
    :param arr: 数值列表
    :return: 众数（空列表返回0，多众数返回第一个）
    """
    if not arr:  # 空列表（无检测结果）
        return 0
    # 统计每个数值的出现频次
    freq_dict = {}
    for num in arr:
        freq_dict[num] = freq_dict.get(num, 0) + 1
    # 找到最大频次
    max_freq = max(freq_dict.values())
    # 提取所有众数（频次等于最大频次），取第一个
    modes = [k for k, v in freq_dict.items() if v == max_freq]
    return modes[0]


# ============================================================
# 新增：读取2.txt的历史累计数据
# ============================================================
def read_history_data(txt_path):
    """
    读取2.txt中的历史累计X（class0总数）、Y（class1总数）
    :param txt_path: 2.txt文件路径
    :return: (历史X, 历史Y)，解析失败/文件不存在返回(0,0)
    """
    if not os.path.exists(txt_path):
        rospy.loginfo(f"2.txt不存在，初始累计值为(0,0)")
        return 0, 0
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # 解析第一行：A街区一共有X+Y人 → 提取总人数
            line1 = lines[0].strip() if len(lines)>=1 else ""
            # 解析第二行：A街区发现Y名非社区人员 → 提取Y
            line2 = lines[1].strip() if len(lines)>=2 else ""
            
            # 提取Y（非社区人员数）
            y_total = 0
            if "A街区发现" in line2 and "名非社区人员" in line2:
                y_str = line2.replace("A街区发现", "").replace("名非社区人员", "")
                y_total = int(y_str) if y_str.isdigit() else 0
            
            # 提取总人数（X+Y），再计算X=总人数-Y
            total_people = 0
            if "A街区一共有" in line1 and "人" in line1:
                total_str = line1.replace("A街区一共有", "").replace("人", "")
                total_people = int(total_str) if total_str.isdigit() else 0
            x_total = total_people - y_total
            
            rospy.loginfo(f"读取历史数据：累计class0(X)={x_total}，累计class1(Y)={y_total}")
            return x_total, y_total
    except Exception as e:
        rospy.logwarn(f"读取2.txt历史数据失败（格式错误/读取异常），初始值设为(0,0)，错误：{e}")
        return 0, 0


# ============================================================
# YOLOv8 letterbox（修复维度问题 + 强制 resize）
# ============================================================
def letterbox(img, new_shape=640, color=(114, 114, 114)):
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

    # Ensure exact shape (in case rounding leads to off-by-one)
    img = img[:new_shape[0], :new_shape[1]]

    return img, r, (dw, dh)


# ============================================================
# 统一 YOLOv8 ONNX 输出格式
# ============================================================
def standardize_output(output):
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


# ============================================================
# NMS 非极大值抑制
# ============================================================
def nms(boxes, scores, iou_thres=0.45):
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


# ============================================================
# YOLO推理核心函数（适配3类模型，过滤第3类 + 面积阈值过滤 + 区域过滤 + 分类颜色）
# ============================================================
def run_inference(frame, session, input_name):
    """
    执行YOLO推理（适配3类模型，仅保留前2类 + 面积阈值过滤 + 仅处理下半区域）
    :param frame: 原始CV2图像
    :param session: ONNX会话
    :param input_name: 模型输入名
    :return: 标注后的图像、检测结果列表、该帧class0数、该帧class1数
    """
    h0, w0 = frame.shape[:2]
    det_img = frame.copy()
    det_list = []
    local_x = 0  # 该帧class0数量
    local_y = 0  # 该帧class1数量

    # 1. 裁剪图像：仅处理下半区域
    crop_y = int(h0 * (1 - DETECT_REGION_RATIO))
    frame_cropped = frame[crop_y:, :]  # 截取下半部分
    h_crop, w_crop = frame_cropped.shape[:2]
    rospy.loginfo(f"裁剪后推理区域尺寸：{h_crop}x{w_crop}（原尺寸：{h0}x{w0}）")

    # 2. 图像预处理（基于裁剪后的图像）
    img_lb, r, (dw, dh) = letterbox(frame_cropped, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_rgb, (2, 0, 1))[None]  # (1, 3, 640, 640)

    # 3. ONNX推理
    try:
        preds = session.run(None, {input_name: img_input})
        preds = standardize_output(preds)
    except Exception as e:
        rospy.logerr(f"推理失败: {e}")
        return det_img, det_list, local_x, local_y

    # 4. 解析检测结果（适配3类模型）
    num_classes = preds.shape[1] - 4
    rospy.loginfo(f"模型输出总类别数: {num_classes}（预期3类）")
    
    # 若模型类别数不是3，给出提示但不终止
    if num_classes != 3:
        rospy.logwarn(f"警告：模型类别数({num_classes})≠3，请确认模型是否正确！")

    boxes = preds[:, :4]
    scores = preds[:, 4:]  # 3类的置信度得分
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]

    # 5. 双重过滤：置信度 + 仅保留前2类（过滤第3类）
    mask = (confs >= CONF_THRESH) & (cls_ids != IGNORE_CLASS_ID)
    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]

    if len(boxes) == 0:
        rospy.loginfo("未检测到有效目标（前2类）")
        return det_img, det_list, local_x, local_y

    # 6. 坐标转换（xywh -> xyxy，还原到裁剪后的图像尺寸）
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2 - dw
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2 - dh
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2 - dw
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2 - dh
    xyxy /= r  # 还原缩放
    
    # 边界裁剪（裁剪后的图像范围内）
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w_crop)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h_crop)

    # 7. NMS非极大值抑制
    keep = nms(xyxy, confs, IOU_THRESH)
    xyxy, confs, cls_ids = xyxy[keep], confs[keep], cls_ids[keep]

    # 8. 处理检测结果（仅输出前2类的0/1 + 面积阈值过滤 + 还原到原图坐标）
    for (x1, y1, x2, y2), sc, cid in zip(xyxy.astype(int), confs, cls_ids):
        # 还原到原始图像的坐标（加上裁剪的y偏移）
        x1_org = x1
        y1_org = y1 + crop_y
        x2_org = x2
        y2_org = y2 + crop_y
        
        # 计算检测框面积（基于原图坐标）
        bbox_area = (x2_org - x1_org) * (y2_org - y1_org)
        # 过滤面积小于阈值的框
        if bbox_area < BBOX_AREA_THRESHOLD:
            rospy.loginfo(f"检测框面积{bbox_area}小于阈值{BBOX_AREA_THRESHOLD}，跳过计数和显示")
            continue
        
        output_val = int(cid)  # 仅0/1（前2类）
        class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
        
        # 统计该帧的class0和class1数量
        if cid == 0:
            local_x += 1
        elif cid == 1:
            local_y += 1
        
        # 打印结果
        rospy.loginfo(f"检测到有效目标：类别ID={cid} → 输出值={output_val}，置信度={sc:.2f}，框面积={bbox_area}，坐标=({x1_org},{y1_org})-({x2_org},{y2_org})")

        # 构建检测结果列表
        det_list.append({
            "class_id": int(cid),
            "output_value": output_val,
            "class_name": class_name,
            "confidence": float(sc),
            "bbox": [int(x1_org), int(y1_org), int(x2_org), int(y2_org)],
            "bbox_area": bbox_area
        })

        # 获取该类别的颜色
        box_color = CLASS_COLORS.get(cid, (0, 255, 0))  # 默认绿色
        # 绘制检测框和文字（在原图上）
        cv2.rectangle(det_img, (x1_org, y1_org), (x2_org, y2_org), box_color, 2)
        cv2.putText(det_img, 
                    f"{class_name}({output_val}) {sc:.2f} Area:{bbox_area}",
                    (x1_org, y1_org-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, box_color, 2)

    # 绘制区域分割线（可视化识别区域）
    cv2.line(det_img, (0, crop_y), (w0, crop_y), (255, 255, 0), 2)
    cv2.putText(det_img, 
                f"Detection Region (Lower {int(DETECT_REGION_RATIO*100)}%)",
                (10, crop_y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)

    return det_img, det_list, local_x, local_y


# ============================================================
# ROS图像回调函数（核心处理逻辑）
# ============================================================
def image_callback(msg):
    global count, saved_files, frame_counts, current_run_imgs
    
    # 达到捕获数量后停止处理
    if count >= CAPTURE_NUM:
        return
    
    try:
        # 1. 将ROS Image消息转换为CV2图像
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        h0, w0 = frame.shape[:2]
        rospy.loginfo(f"\n[{count + 1}/{CAPTURE_NUM}] 收到相机图像 - 尺寸: {h0}x{w0}")

        # 2. 执行YOLO推理（获取该帧的class0/1数量）
        det_img, det_list, frame_x, frame_y = run_inference(frame, session, input_name)

        # 3. 保存标注后的图片（带时间戳，避免覆盖）
        os.makedirs(SAVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        save_path = os.path.join(SAVE_DIR, f"detected_{timestamp}.jpg")
        cv2.imwrite(save_path, det_img)
        rospy.loginfo(f"标注图片已保存: {save_path}")

        # 新增：记录本次运行生成的图片路径
        current_run_imgs.append(save_path)

        # 4. 记录该帧的数量统计结果
        frame_counts.append((frame_x, frame_y))
        rospy.loginfo(f"第{count+1}帧统计：class0={frame_x}，class1={frame_y}")

        # 5. 更新计数和保存列表
        count += 1
        saved_files.append({
            "img_path": save_path,
            "detections": det_list,
            "frame_x": frame_x,  # 记录该帧class0数
            "frame_y": frame_y   # 记录该帧class1数
        })

        # 6. 打印检测结果
        if det_list:
            rospy.loginfo(f"检测结果汇总：{det_list}")
        else:
            rospy.loginfo("本次无有效检测结果（仅前2类且面积达标且在下半区域）")

    except Exception as e:
        rospy.logerr(f"第{count + 1}帧处理失败: {e}")


# ============================================================
# 保留本次匹配众数的1张图片，仅删除本次生成的多余图片（历史图片不删）
# ============================================================
def keep_only_mode_image(save_dir, saved_files, current_run_imgs, x_mode, y_mode):
    """
    仅保留本次运行中与众数匹配的第一张图片，仅删除本次生成的其余图片（历史图片不删）
    :param save_dir: 图片保存目录
    :param saved_files: 本次保存的文件列表（含每帧信息）
    :param current_run_imgs: 本次运行生成的所有图片路径
    :param x_mode: class0众数
    :param y_mode: class1众数
    """
    # 1. 找到本次运行中与众数匹配的第一张图片路径
    keep_img_path = None
    for file_info in saved_files:
        if file_info["frame_x"] == x_mode and file_info["frame_y"] == y_mode:
            keep_img_path = file_info["img_path"]
            break
    
    # 2. 若无匹配的，保留本次最后一张图片
    if not keep_img_path and saved_files:
        keep_img_path = saved_files[-1]["img_path"]
        rospy.logwarn(f"本次无匹配众数({x_mode},{y_mode})的图片，保留本次最后一张：{keep_img_path}")
    
    # 3. 仅删除本次生成的多余图片（历史图片不处理）
    if keep_img_path and current_run_imgs:
        for img_path in current_run_imgs:
            if img_path != keep_img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    rospy.loginfo(f"删除本次多余图片: {img_path}")
                except Exception as e:
                    rospy.logerr(f"删除图片失败 {img_path}: {e}")
        rospy.loginfo(f"本次保留的图片: {keep_img_path}")
    else:
        rospy.logwarn("本次无检测图片可保留")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    # 1. 初始化ROS节点
    rospy.init_node("yolo_onnx_camera_detector", anonymous=False)

    # 2. 加载ONNX模型
    try:
        providers = ["CPUExecutionProvider"]
        # 如需CUDA加速，取消下面注释（需安装onnxruntime-gpu）
        # if rospy.get_param("~use_cuda", False):
        #     providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        session = InferenceSession(MODEL_PATH, providers=providers)
        input_name = session.get_inputs()[0].name
        rospy.loginfo(f"成功加载模型: {MODEL_PATH}")
        rospy.loginfo(f"模型输入名: {input_name}，输入形状: {session.get_inputs()[0].shape}")
    except Exception as e:
        rospy.logerr(f"加载模型失败: {e}")
        exit(1)

    # 3. 打印配置信息
    rospy.loginfo("="*60)
    rospy.loginfo("YOLO ONNX 摄像头识别系统（适配3类模型，仅用前2类 + 面积阈值过滤 + 区域过滤 + 分类颜色）")
    rospy.loginfo("="*60)
    rospy.loginfo(f"摄像头话题: {CAMERA_TOPIC}")
    rospy.loginfo(f"捕获帧数: {CAPTURE_NUM}")
    rospy.loginfo(f"识别区域: 仅下半{int(DETECT_REGION_RATIO*100)}%区域")
    rospy.loginfo(f"模型路径: {MODEL_PATH}")
    rospy.loginfo(f"保存目录: {SAVE_DIR}")
    rospy.loginfo(f"置信度阈值: {CONF_THRESH}")
    rospy.loginfo(f"识别框面积阈值: {BBOX_AREA_THRESHOLD} 像素")
    rospy.loginfo(f"类别颜色配置: class0={CLASS_COLORS[0]}(BGR), class1={CLASS_COLORS[1]}(BGR)")
    rospy.loginfo(f"忽略第{IGNORE_CLASS_ID}类，仅使用前{len(CLASS_NAMES)}类")
    rospy.loginfo(f"TXT文件名: {TXT_FILE_NAME}")
    rospy.loginfo("="*60)

    # 4. 订阅摄像头话题
    rospy.Subscriber(CAMERA_TOPIC, Image, image_callback, queue_size=1, buff_size=2**24)

    # 5. 循环等待（延长运行时间，确保捕获足够帧数）
    rate = rospy.Rate(10)
    start_time = time.time()
    # 条件：未退出ROS + 未捕获足够帧数 + 运行时间<20秒（避免无限等待）
    while not rospy.is_shutdown() and count < CAPTURE_NUM and (time.time() - start_time < 20):
        rate.sleep()

    # 6. 打印每帧统计结果
    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("每帧统计结果：")
    for i, item in enumerate(saved_files, 1):
        rospy.loginfo(f"  第{i}帧: 图片={item['img_path']} | class0={item['frame_x']} | class1={item['frame_y']}")
    
    # 7. 计算众数
    if frame_counts:
        x_list = [x for x, y in frame_counts]  # 提取所有帧的class0数量
        y_list = [y for x, y in frame_counts]  # 提取所有帧的class1数量
        x_mode = get_mode(x_list)              # class0数量的众数
        y_mode = get_mode(y_list)              # class1数量的众数
        rospy.loginfo(f"\n10帧统计众数：class0众数={x_mode}，class1众数={y_mode}")
    else:
        x_mode = 0
        y_mode = 0
        rospy.loginfo("\n无有效检测帧，众数默认设为0")

    # 8. 核心修改：读取历史数据 + 叠加本次众数 + 写入2.txt
    txt_file_path = os.path.join(SAVE_DIR, TXT_FILE_NAME)
    try:
        # 8.1 读取历史累计X、Y
        history_x, history_y = read_history_data(txt_file_path)
        
        # 8.2 叠加本次众数（本次X=x_mode，本次Y=y_mode）
        new_total_x = history_x + x_mode
        new_total_y = history_y + y_mode
        total_people = new_total_x + new_total_y  # 总人数
        
        # 8.3 按指定格式写入（覆盖模式，而非追加）
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(f"A街区一共有{total_people}人\n")
            f.write(f"A街区发现{new_total_y}名非社区人员\n")
        
        rospy.loginfo(f"\n累计结果已保存至rA.txt: {txt_file_path}")
        rospy.loginfo(f"本次叠加：class0众数{x_mode} → 累计class0={new_total_x}；class1众数{y_mode} → 累计class1={new_total_y}")
        rospy.loginfo(f"rA.txt内容：")
        rospy.loginfo(f"  A街区一共有{total_people}人")
        rospy.loginfo(f"  A街区发现{new_total_y}名非社区人员")
    except Exception as e:
        rospy.logerr(f"写入rA.txt失败: {e}")

    # 9. 仅保留本次的1张图片，仅删除本次多余图片（历史图片不删）
    rospy.loginfo("\n开始清理本次多余图片...")
    keep_only_mode_image(SAVE_DIR, saved_files, current_run_imgs, x_mode, y_mode)

    rospy.loginfo("="*60)

    # 10. 退出
    rospy.signal_shutdown("处理完成，退出节点")

