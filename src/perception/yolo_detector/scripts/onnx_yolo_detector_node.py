#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import onnxruntime as ort
import time
import json
import threading

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


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
# NMS
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
# 主节点 —— 实时推理线程 + 最新帧覆盖（0 延迟）
# ============================================================
class YoloOnnxDetector:
    def __init__(self):
        rospy.init_node("yolo_onnx_detector", anonymous=False)

        # ---- 参数 ----
        self.model_path = rospy.get_param("~model_path")
        self.imgsz = rospy.get_param("~img_size", 640)
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.class_names = rospy.get_param("~class_names")

        providers = ["CPUExecutionProvider"]
        if rospy.get_param("~use_cuda", False):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # ---- ONNX 模型 ----
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # ---- 最新图像缓存 ----
        self.latest_frame = None
        self.lock = threading.Lock()

        # ---- ROS I/O ----
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image,
                         self.image_callback,
                         queue_size=1, buff_size=2**24)

        self.pub_img = rospy.Publisher("/yolo/detected_image", Image, queue_size=1)
        self.pub_json = rospy.Publisher("/yolo/detections", String, queue_size=10)

        # ---- 启动推理线程 ----
        threading.Thread(target=self.inference_loop, daemon=True).start()

        rospy.loginfo("YOLO ONNX Realtime Detector Ready (Thread Mode).")

    # ============================================================
    # callback：只保存最新帧，不阻塞
    # ============================================================
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            self.latest_frame = frame

    # ============================================================
    # 推理线程：永远处理最新帧（完全实时）
    # ============================================================
    def inference_loop(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                det_img, det_list = self.run_inference(frame)

                # 发布图像
                msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")
                self.pub_img.publish(msg)

                # =====================================================
                # 发布结构化 JSON（兼容你的 C++ 解析器）
                # =====================================================
                header = {
                    "frame_id": "camera",
                    "stamp": time.time()
                }

                json_msg = {
                    "header": header,
                    "detections": det_list
                }

                self.pub_json.publish(json.dumps(json_msg, ensure_ascii=False))
                # =====================================================

            rate.sleep()

    # ============================================================
    # YOLO 推理流程
    # ============================================================
    def run_inference(self, frame):
        h0, w0 = frame.shape[:2]

        # ---- letterbox ----
        img_lb, r, (dw, dh) = letterbox(frame, self.imgsz)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.
        img_input = np.transpose(img_rgb, (2, 0, 1))[None]

        # ---- ONNX 推理 ----
        preds = self.session.run(None, {self.input_name: img_input})
        preds = standardize_output(preds)

        num_classes = preds.shape[1] - 4
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]

        # ---- 置信度过滤 ----
        mask = confs >= self.conf_thres
        boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]

        if len(boxes) == 0:
            return frame, []

        # ---- xywh → xyxy ----
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2 - dw
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2 - dh
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2 - dw
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2 - dh
        xyxy /= r

        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w0)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h0)

        # ---- NMS ----
        keep = nms(xyxy, confs, self.iou_thres)
        xyxy, confs, cls_ids = xyxy[keep], confs[keep], cls_ids[keep]

        # ---- 绘制 + JSON ----
        det_img = frame.copy()
        det_list = []

        for (x1, y1, x2, y2), sc, cid in zip(xyxy.astype(int), confs, cls_ids):
            name = self.class_names[cid]

            det_list.append({
                "class_id": int(cid),
                "class_name": name,
                "confidence": float(sc),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(det_img, f"{name} {sc:.2f}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

        return det_img, det_list


if __name__ == "__main__":
    detector = YoloOnnxDetector()
    rospy.spin()

