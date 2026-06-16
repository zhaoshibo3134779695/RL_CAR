#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <iostream>
#include <sys/stat.h>
#include <ros/package.h>

/************************************************************************************************
    cmd_send("cmd");
    close_terminal("cmd");
    kill_node("cmd");
    move_to(target_x, target_y, max_time);
    turn_to(target_yaw, max_time);
    wait(delay_time);
    ROS_INFO("");
    nav_send(target_x, target_y, target_qz, target_qw);

    system("rosnode kill /node_name");
    system("rosnode kill -a");

**********************************************************************************************/


#define POS_TOLERANCE 0.2   // 位置容差，单位：米
#define YAW_TOLERANCE 0.1   // 角度容差，单位：弧度


bool path_exists(const std::string& path)
{
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

std::string pkg_path(const std::string& package_name)
{
    const std::string path = ros::package::getPath(package_name);
    if (path.empty())
    {
        ROS_ERROR("missing ROS package path: %s", package_name.c_str());
    }
    return path;
}

std::string voice_runtime_dir()
{
    return pkg_path("voice_broadcast_tool") + "/runtime/monitor_txt";
}

std::string yolo_runtime_dir()
{
    return pkg_path("yolo_detector") + "/runtime";
}

void cmd_send(std::string cmd)
{
    std::string whole_cmd = "gnome-terminal -- bash -c '" + cmd + "; exec bash'";
    int ret = system(whole_cmd.c_str());
    if (ret != 0)
        ROS_WARN("cmd_send failed: %s", cmd.c_str());
}

void close_terminal(std::string cmd)
{
    std::string whole_cmd = "pkill -f '"+ cmd +"'";
    int ret = system(whole_cmd.c_str());
    if (ret != 0)
        ROS_WARN("close_terminal failed or no process matched: %s", cmd.c_str());
}

void close_terminal_2(std::string cmd)
{
    std::string whole_cmd = "pkill -2 -f '"+ cmd +"'";
    int ret = system(whole_cmd.c_str());
    if (ret != 0)
        ROS_WARN("close_terminal_2 failed or no process matched: %s", cmd.c_str());
}

void wait(double delay_time)
{
    if(ros::ok())
        ros::Duration(delay_time).sleep();
}

void kill_node(std::string node_name)
{
    std::string whole_cmd = "rosnode kill " + node_name;
    int ret = system(whole_cmd.c_str());
    if (ret != 0)
        ROS_WARN("kill_node failed: %s", node_name.c_str());
}

void stop()
{
    static ros::NodeHandle nh;
    static bool init = false;
    static ros::Publisher twist_pub = nh.advertise<geometry_msgs::Twist>("/move_vision", 1);
    if(!init)
    {
        wait(1.0);
        init = true;
    }
    static geometry_msgs::Twist twist_msg;

    twist_msg.linear.x = 0.0;
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;

    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = 0.0; 

    twist_pub.publish(twist_msg);
    
}

void stop_flag_set()
{
    ros::param::set("/stop_flag", 1);
    wait(0.3);
    stop();
    ROS_INFO("I have set 111111111111111111111111111");
}

void stop_flag_clear()
{
    ros::param::set("/stop_flag", 0);
    ROS_INFO("I have set 000000000000000000000000000");
}

void stop_and_wait(double wait_time)
{
    ROS_INFO("stop_and_wait %1f s",wait_time);
    stop_flag_set();
    stop();
    wait(wait_time);
    stop_flag_clear();
}

bool check(double target_x, double target_y)
{
    static tf::TransformListener listener; 
    tf::StampedTransform transform;
    try
    {
        // 等待 TF 数据，最多 0.5 秒
        if (!listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(0.5)))
        {
            ROS_WARN("TF not available: map -> base_link");
            return false;
        }

        // 从 tf 中获取 map → base_link 的变换
        listener.lookupTransform("map", "base_link", ros::Time(0), transform);

        // 当前位姿
        double current_x = transform.getOrigin().x();
        double current_y = transform.getOrigin().y();
        
        // 计算位置差
        double dx = target_x - current_x;
        double dy = target_y - current_y;
        double dist = std::sqrt(dx * dx + dy * dy);

        ROS_INFO("dist is %.2f", dist);
        return (dist <= POS_TOLERANCE);
    }
    catch (tf::TransformException &ex)
    {
        ROS_WARN("TF Exception: %s", ex.what());
        return false;
    }
}

bool check2(double target_yaw)
{
    static tf::TransformListener listener; // 静态，避免每次调用都重新构造
    tf::StampedTransform transform;
	target_yaw = target_yaw * 0.0174533;
    try
    {
        // 等待 TF 数据，最多 0.5 秒
        if (!listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(0.5)))
        {
            ROS_WARN("TF not available: map -> base_link");
            return false;
        }

        // 从 tf 中获取 map → base_link 的变换
        listener.lookupTransform("map", "base_link", ros::Time(0), transform);

        // 当前位姿
        double roll, pitch, yaw;
        transform.getBasis().getRPY(roll, pitch, yaw);

        // 计算角度差
        double yaw_diff = std::fabs(target_yaw - yaw);
        if (yaw_diff > M_PI) yaw_diff = 2 * M_PI - yaw_diff; // 归一化到 [0, π]

        // 判断是否在容差范围内
        return (yaw_diff <= YAW_TOLERANCE);
        
    }
    catch (tf::TransformException &ex)
    {
        ROS_WARN("TF Exception: %s", ex.what());
        return false;
    }
}

void move_to(double target_x, double target_y, double max_time)
{
    static int m_n=1;
    ros::Time start = ros::Time::now();
    ros::Duration timeout(max_time);
    static ros::Rate rate(10);
    while (ros::ok() && ros::Time::now() - start < timeout)
    {
        ROS_INFO("this is move to %d",m_n);
        if (check(target_x, target_y))
            break;
        rate.sleep();
    }
    m_n++;
    stop_flag_set();
    //stop_and_wait(1.0);
}

void move_to_ns(double target_x, double target_y, double max_time)
{
    static int m_n=1;
    ros::Time start = ros::Time::now();
    ros::Duration timeout(max_time);
    static ros::Rate rate(10);
    while (ros::ok() && ros::Time::now() - start < timeout)
    {
        ROS_INFO("this is move_to_ns %d",m_n);
        if (check(target_x, target_y))
            break;
        rate.sleep();
    }
    m_n++;
}

void turn_to(double target_yaw, double max_time)
{
    static int m_n=1;
    ros::Time start = ros::Time::now();
    ros::Duration timeout(max_time);
    static ros::Rate rate(10);
    while (ros::ok() && ros::Time::now() - start < timeout)
    {
        ROS_INFO("this is move to %d",m_n);
        if (check2(target_yaw))
            break;
        rate.sleep();
    }
    m_n++;
    //stop_and_wait(1.0);
    stop_flag_set();
    wait(1.0);
    
}

void nav_send(double target_x, double target_y, double target_qz, double target_qw)
{
    static ros::NodeHandle nh;
    static ros::Publisher goal_pub = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1);

    ros::Duration(1.0).sleep(); // 等待订阅建立

    geometry_msgs::PoseStamped goal;
    goal.header.frame_id = "map";
    goal.header.stamp = ros::Time::now();

    goal.pose.position.x = target_x;
    goal.pose.position.y = target_y;
    goal.pose.position.z = 0.0;

    goal.pose.orientation.x = 0.0;
    goal.pose.orientation.y = 0.0;
    goal.pose.orientation.z = target_qz;
    goal.pose.orientation.w = target_qw;

    goal_pub.publish(goal);

    ROS_INFO("Goal sent to: (%.2lf, %.2lf, %.2lf, %.2lf)", target_x, target_y, target_qz, target_qw);
}

void set_left()
{
    ros::param::set("/stop_flag", 2);
    wait(0.3);
    stop();
}

void turn_left(double delay_time)
{
    ros::param::set("/stop_flag", 2);
    wait(0.3);
    stop();
    wait(delay_time);
}

void go_forward(double delay_time)
{
    ros::param::set("/stop_flag", 3);
    wait(0.3);
    stop();
    wait(delay_time);
}

geometry_msgs::PoseWithCovarianceStamped tf_save()
{
    
    static ros::NodeHandle nh;
    static geometry_msgs::PoseWithCovarianceStamped pose_msg;
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    wait(1.0);
    //
    double x = 0.0;
    double y = 0.0;
    double qz = 0.0;
    double qw = 0.0;
    try
    {
        listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(2.0));
        listener.lookupTransform("map", "base_link", ros::Time(0), transform);

        x  = transform.getOrigin().x();
        y  = transform.getOrigin().y();
        qz = transform.getRotation().z();
        qw = transform.getRotation().w();

        ROS_INFO("x: %.3f, y: %.3f, qz: %.3f, qw: %.3f", x, y, qz, qw);
    }
    catch (tf::TransformException &ex)
    {
        ROS_WARN("TF读取失败: %s", ex.what());
    }
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = x;
    pose_msg.pose.pose.position.y = y;
    pose_msg.pose.covariance[0] = 0.25;
    pose_msg.pose.covariance[6 * 1 + 1] = 0.25;
    pose_msg.pose.covariance[6 * 5 + 5] = 0.06853891945200942;
    pose_msg.pose.pose.orientation.z = qz;
    pose_msg.pose.pose.orientation.w = qw;
    return pose_msg;
}

void tf_reset(geometry_msgs::PoseWithCovarianceStamped pose_msg)
{
    static ros::NodeHandle nh;
    static ros::Publisher initial_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 10); 
    wait(1.0);
    pose_msg.header.stamp = ros::Time::now();
    initial_pose_pub.publish(pose_msg);
    wait(1.0);
}

void publishTargetYaw(double yaw_degrees)
{
    static ros::Publisher pub = ros::NodeHandle().advertise<std_msgs::Float64>("/target_yaw", 10);
    
    std_msgs::Float64 msg;
    msg.data = yaw_degrees;
    pub.publish(msg);
    
    ROS_INFO("Published target yaw: %.2f degrees", yaw_degrees);
}

void move_to(double, double, double max_time = 60.0);
void move_to_ns(double target_x, double target_y, double max_time = 60.0);
void turn_to(double, double max_time = 60.0);
void nav_send(double, double, double, double);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "task_schedule");
    ros::NodeHandle nh;
    const std::string fire_pkg_dir = pkg_path("fire_detection_tool");
    const std::string waste_pkg_dir = pkg_path("waste_classification_tool");
    const std::string voice_pkg_dir = pkg_path("voice_broadcast_tool");
    const std::string yolo_pkg_dir = pkg_path("yolo_detector");
    const std::string nav_launch = pkg_path("nav_demo") + "/launch/nav777.launch";
    const std::string voice_monitor_dir = voice_runtime_dir();
    const std::string yolo_runtime = yolo_runtime_dir();

    cmd_send("rm -rf -- " + fire_pkg_dir + "/captured_images/*");
    cmd_send("rm -rf -- " + fire_pkg_dir + "/title_crops/*");
    cmd_send("rm -rf -- " + yolo_runtime + "/ddc_txt/*");
    cmd_send("rm -rf -- " + yolo_runtime + "/r/*");
    geometry_msgs::PoseWithCovarianceStamped pose_msg;
    stop_flag_set();
    stop_flag_clear();
    go_forward(0.0);
    turn_left(0.0);
    
    ros::param::set("/line_vel", 1.3);
    ros::param::set("/argu_vel", 1.5);
    ros::param::set("/auto_argu", 1);
    
    stop();
	
    cmd_send("rosrun car_bringup newt.py");
    ROS_INFO("newt.py     start");
    
    cmd_send("roslaunch  orbbec_camera dabai_dcw2.launch");
    ROS_INFO("dabai_dcw2.launch     start");
    
    cmd_send("rosrun yolo_detector camera_send.py /dev/video0");
    ROS_INFO("camera_send.py     start");
    
    cmd_send("rqt_image_view");
    ROS_INFO("rqt_image_view     start");
    
    cmd_send("cd " + voice_pkg_dir + " && python3 piper_monitor.py");
    ROS_INFO("piper_monitor.py     start");
    
    cmd_send("rosrun task_schedule vel_ctrl");
    ROS_INFO("vel_ctrl     start");
    
    cmd_send("rosrun task_schedule tf_info");
    ROS_INFO("tf_info           start");
    
    cmd_send("roslaunch xy_path_follower yaw_pid_controller.launch");
    ROS_INFO("yaw_pid_controller.launch     start");
    
    if (path_exists(nav_launch))
    {
        cmd_send("roslaunch nav_demo nav777.launch");
        ROS_INFO("nav777.launch     start");
    }
    else
    {
        ROS_ERROR("missing launch file: %s", nav_launch.c_str());
    }
    
    wait(5.0);
    /////////////////////////////////////////////////////////////////////////////
    
    cmd_send("roslaunch xy_path_follower xy_path_follower.launch");
    ROS_INFO("xy_path_follower.launch     start");
    publishTargetYaw(0.0);
    
    //第一个红绿灯
    move_to(1.05, 0);
   
    cmd_send("roslaunch yolo_detector onnx_yolo_detector.launch");
    ROS_INFO("onnx_yolo_detector.launch     start");
    cmd_send("roslaunch yolo_detector yolo_json_parser.launch");
    ROS_INFO("yolo_json_parser.launch     start");
    stop_flag_set();
    
    move_to_ns(2.0,0);
    close_terminal("roslaunch yolo_detector onnx_yolo_detector.launch");
    ROS_INFO("onnx_yolo_detector.launch     end");
    close_terminal("roslaunch yolo_detector yolo_json_parser.launch");
    ROS_INFO("yolo_json_parser.launch     end");

    //A街区1
    move_to(2.63, 0);
    publishTargetYaw(90.0);
    wait(3.0);
    cmd_send("rosrun yolo_detector rA.py");
    ROS_INFO("rA.py     start");
    cmd_send("rosrun yolo_detector weitingshibieA.py");
    ROS_INFO("weitingshibieA.py     start");
    stop_and_wait(10.0);
	publishTargetYaw(0.0);
	stop_and_wait(3.0);
	
    //禁止通行识别，播报1
    move_to(3.3, 0);
    publishTargetYaw(90.0);
    wait(3.0);
    cmd_send("rosrun yolo_detector bzshibie.py");
    ROS_INFO("bzshibie.py     start");
    stop_and_wait(5.0);

    //A街区2
    move_to(3.35, 0.55);
    publishTargetYaw(180.0);
    wait(3.0);
    cmd_send("rosrun yolo_detector rA.py");
    ROS_INFO("rA.py     start");
    cmd_send("rosrun yolo_detector weitingshibieA.py");
    ROS_INFO("weitingshibieA.py     start");
    stop_and_wait(10.0);

	//B街区1
	move_to(2.65, 1.15);
    publishTargetYaw(90.0);
    wait(3.0);
    cmd_send("rosrun yolo_detector rB.py");
    ROS_INFO("rB.py     start");
    cmd_send("rosrun yolo_detector weitingshibieB.py");
    ROS_INFO("weitingshibieB.py     start");
    stop_and_wait(10.0);

    //A街区3
    move_to(2.40, 1.15);
    publishTargetYaw(0.0);
    wait(0.5);
    publishTargetYaw(-90.0);
    wait(3.0);
    cmd_send("rosrun yolo_detector rA.py");
    ROS_INFO("rA.py     start");
    cmd_send("rosrun yolo_detector weitingshibieA.py");
    ROS_INFO("weitingshibieA.py     start");
    stop_and_wait(10.0);

    //大楼1，播报2
    move_to(1.95, 1.05);
    publishTargetYaw(180.0);
    stop_flag_set();
    wait(3.0);
    cmd_send("cd " + fire_pkg_dir + " && python3 fire_detection_pipeline1.py");
    ROS_INFO("fire_detection_pipeline1.py     start");
    stop_and_wait(5.0);

    //大楼3，播报3
    move_to(1.8, 1.75);
    stop_flag_set();
    wait(5.0);
    cmd_send("rm -rf -- " + fire_pkg_dir + "/captured_images/*");
    cmd_send("rm -rf -- " + fire_pkg_dir + "/title_crops/*");
    cmd_send("cd " + fire_pkg_dir + " && python3 fire_detection_pipeline3.py");
    ROS_INFO("fire_detection_pipeline3.py     start");
    stop_flag_set();
    wait(5.0);

    //B街区2，播报4
    publishTargetYaw(90.0);
    wait(0.5);
    publishTargetYaw(0.0);
    stop_flag_set();
    wait(3.0);
    cmd_send("rosrun yolo_detector rB.py");
    ROS_INFO("rB.py     start");
    cmd_send("rosrun yolo_detector weitingshibieB.py");
    ROS_INFO("weitingshibieB.py     start");
    stop_flag_set();
    wait(10.0);
    cmd_send("rosrun yolo_detector summary_rAB.py");
    ROS_INFO("summary_rAB.py     start");


    //第二个红绿灯
    publishTargetYaw(90.0);
    stop_flag_set();
    wait(3.0);
    cmd_send("roslaunch yolo_detector onnx_yolo_detector.launch");
    ROS_INFO("onnx_yolo_detector.launch     start");
    cmd_send("roslaunch yolo_detector yolo_json_parser.launch");
    ROS_INFO("yolo_json_parser.launch     start");
    
    move_to_ns(1.8, 2.90);
    close_terminal("roslaunch yolo_detector onnx_yolo_detector.launch");
    ROS_INFO("onnx_yolo_detector.launch     end");
    close_terminal("roslaunch yolo_detector yolo_json_parser.launch");
    ROS_INFO("yolo_json_parser.launch     end");
    
    move_to_ns(1.65, 3.4);
    publishTargetYaw(180.0);

    //车牌识别1，播报5
    move_to(0.65, 3.4);
    cmd_send("rosrun yolo_detector chepaishibie1.py");
    ROS_INFO("chepaishibie1.py     start");
    stop_and_wait(5.0);

    //车牌识别2，播报6
    move_to(0.55, 2.8);
    cmd_send("rosrun yolo_detector chepaishibie2.py");
    ROS_INFO("chepaishibie2.py     start");
    stop_and_wait(5.0);

    //垃圾识别，播报7
    move_to(0.55, 2.6);
    stop_flag_set();
    publishTargetYaw(90.0);
    wait(0.5);
    publishTargetYaw(0.0);
    wait(3.0); 
    cmd_send("cd " + waste_pkg_dir + " && python3 capture_and_infer.py");
    ROS_INFO("capture_and_infer.py     start");
    stop_and_wait(5.0);

    //车牌识别3，播报8
    move_to(0.55, 2.15);
    publishTargetYaw(90.0);
    wait(0.5);
    publishTargetYaw(180.0);
    stop_flag_set();
    wait(3.0);
    cmd_send("rosrun yolo_detector chepaishibie3.py");
    ROS_INFO("chepaishibie3.py     start");
    stop_and_wait(5.0);
    
	close_terminal("roslaunch xy_path_follower xy_path_follower.launch");
    ROS_INFO("xy_path_follower.launch     end");
    
    //大楼2，播报9
    cmd_send("roslaunch xy_path_follower xy_path_follower2.launch");
    ROS_INFO("xy_path_follower.launch     start");
    publishTargetYaw(90.0);
    wait(0.5);
    publishTargetYaw(0.0);
    stop_and_wait(3.0);
    move_to(0.624, 1.55);
    stop_flag_set();
    cmd_send("rm -rf -- " + fire_pkg_dir + "/captured_images/*");
    cmd_send("rm -rf -- " + fire_pkg_dir + "/title_crops/*");
    cmd_send("cd " + fire_pkg_dir + " && python3 fire_detection_pipeline2.py");
    ROS_INFO("fire_detection_pipeline.py     start");
    stop_and_wait(5.0);
    

    //电动车识别
    publishTargetYaw(90.0);
    wait(0.5);
    publishTargetYaw(180.0);
    ros::param::set("/line_vel", 0.4);

    stop_flag_set();
    wait(3.0);
    
    ros::param::set("/stop_flag", 4);
    wait(3.0);
    stop_flag_set();
    
    cmd_send("rosrun yolo_detector ddczuizhong.py");
    ROS_INFO("ddczuizhong.py     start");
    wait(5.0);
    stop_flag_clear();
    
    move_to(0.65, 0.10);
    stop_and_wait(5.0);
    
    ros::param::set("/line_vel", 1.5);
    publishTargetYaw(-90.0);
    wait(0.5);
    publishTargetYaw(0.0);
    stop_and_wait(3.0);
    move_to(0.0, 0.0);
    cmd_send("rosrun yolo_detector summary_ddcAB.py");
    ROS_INFO("summary_ddcAB.py     start");
    close_terminal("roslaunch xy_path_follower xy_path_follower2.launch");
    ROS_INFO("xy_path_follower.launch     end");
    
    ////////////////////////////////////////////////////////////////////////////
    ros::param::set("/auto_argu", 0);
    stop_flag_set();
    wait(0.3);
    stop();
    
    
    close_terminal("roslaunch nav_demo nav777.launch");
    ROS_INFO("nav777.launch     end");
    
    close_terminal("roslaunch xy_path_follower yaw_pid_controller.launch");
    ROS_INFO("yaw_pid_controller.launch     end");
    
    close_terminal("rosrun task_schedule vel_ctrl");
    ROS_INFO("vel_ctrl          end");
    
    close_terminal("rosrun task_schedule tf_info");
    ROS_INFO("tf_info           end");
    
    close_terminal("camera_send.py");
    ROS_INFO("camera_send.py     end");
    
    close_terminal("roslaunch  orbbec_camera dabai_dcw2.launch");
    ROS_INFO("dabai_dcw2.launch     end");
    
    close_terminal("rqt_image_view");
    ROS_INFO("rqt_image_view    end");
    
    close_terminal("rosrun car_bringup newt.py");
    ROS_INFO("newt.py          end");
    wait(60.0);
    
    close_terminal("piper_monitor.py");
    ROS_INFO("piper_monitor.py          end");
    
    return 0;
}
