#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

ros::Publisher twist_pub;

void move_vision_Callback(const geometry_msgs::Twist::ConstPtr& msg)
{
    static int stop_flag = 0;
	static double line_vel = 1.0;
	static double argu_vel = 1.0;
    static int auto_argu = 1;
	
    ros::param::get("/stop_flag", stop_flag);
    ros::param::get("/line_vel", line_vel);
    ros::param::get("/argu_vel", argu_vel);
    ros::param::get("/auto_argu", auto_argu);
    
    geometry_msgs::Twist twist_msg;
    twist_msg.linear.x = 0.0;
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;

    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = 0.0;

    if(auto_argu)
    {
        twist_msg.angular.x = (*msg).angular.x;
        twist_msg.angular.y = (*msg).angular.y;
        twist_msg.angular.z = (*msg).angular.z;
    }
    
    if(stop_flag == 0)
    {
        ROS_INFO("got 0");
        twist_msg = *msg;
    }
    else if(stop_flag == 1)
    {
        ROS_INFO("got 1");
    }
    else if(stop_flag == 2)
    {
        ROS_INFO("got 2");
        twist_msg.angular.z = 0.2;
    }
    else if(stop_flag == 3)
    {
        ROS_INFO("got 3");
        twist_msg.linear.x = 0.2;
    }
    else if(stop_flag == 4)
    {
        ROS_INFO("got 4");
        twist_msg.linear.y = -0.2;
    }
    else if(stop_flag == 5)
    {
        ROS_INFO("got 5");
        twist_msg.linear.y = 0.2;
    }
    
    twist_msg.linear.x *= line_vel;
	twist_msg.linear.y *= line_vel;
	twist_msg.linear.z *= line_vel;

	twist_msg.angular.x *= argu_vel;
	twist_msg.angular.y *= argu_vel;
	twist_msg.angular.z *= argu_vel;
	
    twist_pub.publish(twist_msg);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "vel_ctrl");
    ros::NodeHandle nh;
    ros::param::set("/stop_flag", 0);
    twist_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    ros::Subscriber pose_sub = nh.subscribe("/move_vision", 2, move_vision_Callback);
    ros::spin();
    return 0;
}
