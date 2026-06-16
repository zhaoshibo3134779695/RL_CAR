#!/usr/bin/env python3
import cv2
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def open_camera(device_path="/dev/video0"):
    # 初始化ROS节点
    rospy.init_node('camera_publisher', anonymous=True)
    # 创建图像发布者，话题名：/camera/image_raw，队列大小：10
    image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    # 创建CVBridge（转换OpenCV图像到ROS图像消息）
    bridge = CvBridge()
    
    # 关键：指定 v4l2 驱动 + 直接传入设备路径（而非数字索引）
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    
    # 强制设置摄像头分辨率（避免驱动兼容问题）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头：{device_path}")
        # 额外排查：打印所有可用摄像头
        print("🔍 系统可用视频设备：")
        import os
        for i in range(10):
            if os.path.exists(f"/dev/video{i}"):
                test_cap = cv2.VideoCapture(f"/dev/video{i}", cv2.CAP_V4L2)
                status = "✅" if test_cap.isOpened() else "❌"
                print(f"  /dev/video{i}: {status}")
                test_cap.release()
        sys.exit(1)
    
    print(f"✅ 成功打开摄像头：{device_path}")
    print("📢 正在发布图像到ROS话题：/camera/image_raw")
    print("💡 按下 'q' 键或关闭窗口退出")
    
    # 设置ROS循环频率（与摄像头FPS匹配）
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面（可能是摄像头被占用）")
            break
        
        # 将OpenCV图像（BGR格式）转换为ROS图像消息并发布
        try:
            ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
            # 设置消息时间戳（可选，建议添加）
            ros_image.header.stamp = rospy.Time.now()
            # 发布图像
            image_pub.publish(ros_image)
        except CvBridgeError as e:
            print(f"❌ 图像转换失败：{e}")
        
        # 显示本地预览画面
        cv2.imshow("Camera Preview", frame)
        
        # 退出条件（兼容不同键盘布局）
        if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]:  # 27是ESC键
            break
        
        # 按设定频率休眠
        rate.sleep()
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("🔌 已关闭摄像头，停止发布图像话题")

if __name__ == "__main__":
    device_path = "/dev/video0"
    if len(sys.argv) > 1:
        device_path = sys.argv[1]
    try:
        open_camera(device_path)
    except rospy.ROSInterruptException:
        print("🔌 ROS节点中断，已关闭摄像头")

