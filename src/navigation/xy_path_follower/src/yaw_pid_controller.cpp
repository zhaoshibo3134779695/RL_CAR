#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class YawPIDController
{
public:
    YawPIDController(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : tf_listener_(tf_buffer_),
          have_target_yaw_(true),
          have_xy_vel_(false),
          prev_error_(0.0),
          integral_(0.0)
    {
        // PID 和其他参数
        pnh.param("kp", kp_, 1.0);                   // 比例系数
        pnh.param("ki", ki_, 0.0);                   // 积分系数
        pnh.param("kd", kd_, 0.0);                   // 微分系数
        pnh.param("max_ang_vel", max_ang_vel_, 1.0); // 角速度饱和上限（rad/s）

        pnh.param<std::string>("world_frame", world_frame_, "map");
        pnh.param<std::string>("base_frame",  base_frame_,  "base_link");

        double control_rate;
        pnh.param("control_rate", control_rate, 50.0); // Hz

        // 订阅和发布
        sub_xy_vel_ = nh.subscribe("/xy_vel", 1, &YawPIDController::xyVelCallback, this);
        sub_target_yaw_ = nh.subscribe("/target_yaw", 1, &YawPIDController::targetYawCallback, this);

        pub_cmd_vel_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

        last_time_ = ros::Time::now();

        // 定时器：固定频率做控制
        timer_ = nh.createTimer(ros::Duration(1.0 / control_rate), &YawPIDController::controlTimerCallback, this);
    }

private:
    // 角度归一化到 [-pi, pi)
    double normalizeAngleRad(double a)
    {
        while (a >= M_PI)  a -= 2.0 * M_PI;
        while (a < -M_PI)  a += 2.0 * M_PI;
        return a;
    }

    void xyVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        last_xy_vel_ = *msg;
        have_xy_vel_ = true;
    }

    // target_yaw: 单位为“度”
    void targetYawCallback(const std_msgs::Float64::ConstPtr& msg)
    {
        target_yaw_rad_ = msg->data * M_PI / 180.0; // 转弧度
        target_yaw_rad_ = normalizeAngleRad(target_yaw_rad_);
        have_target_yaw_ = true;
    }

    void controlTimerCallback(const ros::TimerEvent& event)
    {
        if (!have_target_yaw_ || !have_xy_vel_)
            return;

        ros::Time now = ros::Time::now();
        double dt = (now - last_time_).toSec();
        if (dt <= 0.0)
        {
            last_time_ = now;
            return;
        }

        double current_yaw = 0.0;
        if (!getCurrentYaw(current_yaw))
        {
            ROS_WARN_THROTTLE(1.0, "YawPIDController: failed to lookup TF %s -> %s",
                              world_frame_.c_str(), base_frame_.c_str());
            last_time_ = now;
            return;
        }

        // 误差 = 目标 - 当前
        double error = normalizeAngleRad(target_yaw_rad_ - current_yaw);

        // PID
        integral_ += error * dt;
        double derivative = (error - prev_error_) / dt;
        double output = kp_ * error + ki_ * integral_ + kd_ * derivative;

        // 限幅
        if (output >  max_ang_vel_) output =  max_ang_vel_;
        if (output < -max_ang_vel_) output = -max_ang_vel_;

        prev_error_ = error;
        last_time_ = now;

        geometry_msgs::Twist cmd = last_xy_vel_; // 保留线速度
        cmd.angular.z = output;                  // 只改角速度
        pub_cmd_vel_.publish(cmd);
    }

    bool getCurrentYaw(double& yaw_out)
    {
        geometry_msgs::TransformStamped transformStamped;
        try
        {
            // world_frame_ -> base_frame_
            transformStamped = tf_buffer_.lookupTransform(
                world_frame_, base_frame_, ros::Time(0), ros::Duration(0.05));
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN_THROTTLE(1.0, "TF exception: %s", ex.what());
            return false;
        }

        tf2::Quaternion q;
        tf2::fromMsg(transformStamped.transform.rotation, q);

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        yaw_out = yaw; // 弧度
        return true;
    }

private:
    // ROS
    ros::Subscriber sub_xy_vel_;
    ros::Subscriber sub_target_yaw_;
    ros::Publisher  pub_cmd_vel_;
    ros::Timer      timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // 状态
    geometry_msgs::Twist last_xy_vel_;
    bool   have_target_yaw_;
    bool   have_xy_vel_;

    double target_yaw_rad_ = 0.0;   // 弧度
    double prev_error_;
    double integral_;
    ros::Time last_time_;

    // 参数
    double kp_, ki_, kd_;
    double max_ang_vel_;
    std::string world_frame_;
    std::string base_frame_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yaw_pid_controller");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    YawPIDController node(nh, pnh);

    ros::spin();
    return 0;
}
