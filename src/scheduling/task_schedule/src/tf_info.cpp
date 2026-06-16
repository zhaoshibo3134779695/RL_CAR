#include <ros/ros.h>
#include <tf/transform_listener.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tf_reader_xy_qz_qw");
    ros::NodeHandle nh;

    tf::TransformListener listener;
    tf::StampedTransform transform;

    ros::Rate rate(10); // 10 Hz

    while (ros::ok())
    {
        try
        {
            listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("map", "base_link", ros::Time(0), transform);

            double x  = transform.getOrigin().x();
            double y  = transform.getOrigin().y();
            double qz = transform.getRotation().z();
            double qw = transform.getRotation().w();

            ROS_INFO("x: %.3f, y: %.3f, qz: %.3f, qw: %.3f", x, y, qz, qw);
        }
        catch (tf::TransformException &ex)
        {
            ROS_WARN("TF读取失败: %s", ex.what());
        }

        rate.sleep();
    }

    return 0;
}
