from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    path = get_package_share_path('raicom_research_assets') / 'config/replay/ekf_wheel_imu.yaml'
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(path)),
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        Node(package='robot_localization', executable='ekf_node', name='research_ekf',
             parameters=[LaunchConfiguration('config'),
                         {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],
             remappings=[('odometry/filtered', '/research/odometry/filtered')]),
    ])
