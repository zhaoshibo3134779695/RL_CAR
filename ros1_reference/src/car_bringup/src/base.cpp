#include "base.h"
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_datatypes.h>
RobotBase::RobotBase(ros::NodeHandle nh,ros::NodeHandle nh_private) :
        nh_(nh),
        nh_private(nh_private),
        linear_velocity_x_(0.0),
        linear_velocity_y_(0.0),
        angular_velocity_z_(0.0),
        last_vel_time_(0),
        max_vel_dt_(0.5),
        imu_timeout_(0.2),
        vel_dt_(0.0),
        x_pos_(0.0),
        y_pos_(0.0),
        heading_(0.0),
        imu_heading_(0.0),
        last_imu_yaw_(0.0),
        imu_angular_velocity_z_(0.0),
        use_imu_yaw_(false),
        imu_initialized_(false) {
    nh_private.param<double>("linear_scale_x", linear_scale_x, 1.0);
    nh_private.param<double>("linear_scale_y", linear_scale_y, 1.0);
    nh_private.param<double>("angular_scale", angular_scale, 1.0);
    nh_private.param<double>("max_vel_dt", max_vel_dt_, 0.5);
    nh_private.param<double>("imu_timeout", imu_timeout_, 0.2);
    nh_private.param<bool>("use_imu_yaw", use_imu_yaw_, false);
    nh_private.param<string>("odom_frame", odom_frame, string("odom"));
    nh_private.param<string>("base_footprint_frame", base_footprint_frame, string("base_footprint"));
    velocity_subscriber_ = nh_.subscribe("/sub_vel", 50, &RobotBase::velCallback, this);
    if (use_imu_yaw_) {
        string imu_topic;
        nh_private.param<string>("imu_topic", imu_topic, string("/imu0"));
        imu_subscriber_ = nh_.subscribe(imu_topic, 100, &RobotBase::imuCallback, this);
    }
    odom_publisher_ = nh_.advertise<nav_msgs::Odometry>("/pub_odom", 50);
//    ROS_ERROR_STREAM("linear_scale_x: "<<linear_scale_x);
//    ROS_ERROR_STREAM("linear_scale_y: "<<linear_scale_y);
//    ROS_ERROR_STREAM("odom_frame: "<<odom_frame);
//    ROS_ERROR_STREAM("base_footprint_frame: "<<base_footprint_frame);
}

void RobotBase::imuCallback(const sensor_msgs::Imu::ConstPtr& imu) {
    const double raw_yaw = tf::getYaw(imu->orientation);
    if (!imu_initialized_) {
        last_imu_yaw_ = raw_yaw;
        imu_heading_ = heading_;
        imu_initialized_ = true;
    } else {
        const double delta_yaw = atan2(
            sin(raw_yaw - last_imu_yaw_), cos(raw_yaw - last_imu_yaw_));
        imu_heading_ += delta_yaw;
        last_imu_yaw_ = raw_yaw;
    }
    imu_angular_velocity_z_ = imu->angular_velocity.z;
    last_imu_time_ = imu->header.stamp.isZero() ? ros::Time::now() : imu->header.stamp;
}

void RobotBase::velCallback(const geometry_msgs::Twist twist) {
//    ROS_INFO("ODOM PUBLISH %.2f,%.2f,%.2f", twist.linear.x, twist.linear.y, twist.angular.z);
    ros::Time current_time = ros::Time::now();
    linear_velocity_x_ = twist.linear.x * linear_scale_x;
    linear_velocity_y_ = twist.linear.y * linear_scale_y;
    const double wheel_angular_velocity = twist.angular.z * angular_scale;
    if (last_vel_time_.isZero()) {
        vel_dt_ = 0.0;
    } else {
        vel_dt_ = (current_time - last_vel_time_).toSec();
        if (vel_dt_ < 0.0 || vel_dt_ > max_vel_dt_) {
            vel_dt_ = 0.0;
        }
    }
    last_vel_time_ = current_time;
    const bool imu_fresh = use_imu_yaw_ && imu_initialized_ &&
        (current_time - last_imu_time_).toSec() <= imu_timeout_;
    const double previous_heading = heading_;
    if (imu_fresh) {
        heading_ = imu_heading_;
        angular_velocity_z_ = imu_angular_velocity_z_;
    } else {
        heading_ += wheel_angular_velocity * vel_dt_;
        angular_velocity_z_ = wheel_angular_velocity;
    }

    // Midpoint heading keeps simultaneous translation and rotation consistent.
    const double integration_heading = 0.5 * (previous_heading + heading_);
    double delta_x = (linear_velocity_x_ * cos(integration_heading) - linear_velocity_y_ * sin(integration_heading)) * vel_dt_; //m
    double delta_y = (linear_velocity_x_ * sin(integration_heading) + linear_velocity_y_ * cos(integration_heading)) * vel_dt_; //m
    //calculate current position of the robot
    x_pos_ += delta_x;
    y_pos_ += delta_y;
    //calculate robot's heading in quaternion angle
    //ROS has a function to calculate yaw in quaternion angle
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(heading_);
    nav_msgs::Odometry odom;
    odom.header.stamp = current_time;
    odom.header.frame_id = odom_frame;
    odom.child_frame_id = base_footprint_frame;
    // robot's position in x,y and z
    odom.pose.pose.position.x = x_pos_;
    odom.pose.pose.position.y = y_pos_;
    odom.pose.pose.position.z = 0.0;
    // robot's heading in quaternion
    odom.pose.pose.orientation = odom_quat;
    odom.pose.covariance[0] = 0.001;
    odom.pose.covariance[7] = 0.001;
    odom.pose.covariance[35] = 0.001;
    // linear speed from encoders
    odom.twist.twist.linear.x = linear_velocity_x_;
    odom.twist.twist.linear.y = linear_velocity_y_;
    odom.twist.twist.linear.z = 0.0;
    odom.twist.twist.angular.x = 0.0;
    odom.twist.twist.angular.y = 0.0;
    // angular speed from encoders
    odom.twist.twist.angular.z = angular_velocity_z_;
    odom.twist.covariance[0] = 0.0001;
    odom.twist.covariance[7] = 0.0001;
    odom.twist.covariance[35] = 0.0001;

    geometry_msgs::TransformStamped odom_transform;
    odom_transform.header.stamp = current_time;
    odom_transform.header.frame_id = odom_frame;
    odom_transform.child_frame_id = base_footprint_frame;
    odom_transform.transform.translation.x = x_pos_;
    odom_transform.transform.translation.y = y_pos_;
    odom_transform.transform.translation.z = 0.0;
    odom_transform.transform.rotation = odom_quat;

    odom_broadcaster_.sendTransform(odom_transform);
    // ROS_INFO("ODOM PUBLISH");
    odom_publisher_.publish(odom);
}
