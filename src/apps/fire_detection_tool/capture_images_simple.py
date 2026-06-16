#!/usr/bin/env python3
# filepath: c:\Users\26233\Desktop\zhsq_waste_classification\capture_images_simple.py
"""
直接修改代码中的配置
"""
import rospy
import cv2
import os
from pathlib import Path
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from runtime_paths import CAPTURE_DIR, ensure_dirs

# ========== 配置区域 ==========
CAMERA_TOPIC = '/camera/color/image_raw'  # 相机话题
SAVE_DIR = str(CAPTURE_DIR)
NUM_IMAGES = 1                      # 捕获数量
FILE_PREFIX = 'capture'             # 文件前缀
IMAGE_FORMAT = 'jpg'                # 图片格式
# ==============================

bridge = CvBridge()
count = 0
saved_files = []

def callback(msg):
    global count, saved_files
    if count >= NUM_IMAGES:
        return
    
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{FILE_PREFIX}_{count + 1}_{timestamp}.{IMAGE_FORMAT}"
        filepath = os.path.join(SAVE_DIR, filename)
        
        cv2.imwrite(filepath, cv_img)
        count += 1
        saved_files.append(filepath)
        print(f"✓ [{count}/{NUM_IMAGES}] {filename}")
        
    except Exception as e:
        print(f"✗ 错误: {e}")

def main():
    global count
    
    ensure_dirs()
    print(f"\n开始从 {CAMERA_TOPIC} 捕获 {NUM_IMAGES} 张图片...")
    print(f"保存到: {SAVE_DIR}\n")
    
    rospy.init_node('image_capture', anonymous=True)
    sub = rospy.Subscriber(CAMERA_TOPIC, Image, callback, queue_size=1)
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and count < NUM_IMAGES:
        rate.sleep()
    
    sub.unregister()
    
    print(f"\n完成！共保存 {len(saved_files)} 张:")
    for f in saved_files:
        print(f"  {f}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断")
