#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接修改代码中的配置
"""
import rospy
import cv2
import os
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from runtime_paths import CAPTURE_DIR, ensure_dirs

# ========== 配置区域 ==========
CAMERA_TOPIC = '/camera/color/image_raw'  # 相机话题
SAVE_DIR = str(CAPTURE_DIR)
NUM_IMAGES = 1                             # 捕获数量
IMAGE_FORMAT = 'jpg'                       # 图片格式（扩展名）
# ==============================

bridge = CvBridge()
count = 0
saved_files = []

def _clear_save_dir(dir_path):
    """删除保存目录中的所有图片文件"""
    try:
        ensure_dirs()
        for name in os.listdir(dir_path):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p) and name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                os.remove(p)
        print("已清空保存目录: {}".format(dir_path))
    except Exception as e:
        print("清空保存目录失败: {}".format(e))

def callback(msg):
    global count, saved_files
    if count >= NUM_IMAGES:
        return
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        # 按序号命名：1.jpg、2.jpg、3.jpg …
        filename = "{}.{}".format(count + 1, IMAGE_FORMAT)
        filepath = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(filepath, cv_img)
        count += 1
        saved_files.append(filepath)
        print("✓ [{}/{}] {}".format(count, NUM_IMAGES, filename))
    except Exception as e:
        print("✗ 错误: {}".format(e))

def main():
    global count
    # 先清空保存目录
    _clear_save_dir(SAVE_DIR)

    print("\n开始从 {} 捕获 {} 张图片...".format(CAMERA_TOPIC, NUM_IMAGES))
    print("保存到: {}\n".format(SAVE_DIR))

    rospy.init_node('image_capture', anonymous=True)
    sub = rospy.Subscriber(CAMERA_TOPIC, Image, callback, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and count < NUM_IMAGES:
        rate.sleep()

    sub.unregister()

    print("\n完成！共保存 {} 张:".format(len(saved_files)))
    for f in saved_files:
        print("  {}".format(f))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断")
