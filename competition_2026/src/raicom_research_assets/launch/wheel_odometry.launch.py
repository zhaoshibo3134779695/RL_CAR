import yaml
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def start(context):
    path = get_package_share_path('raicom_research_assets') / 'config/competition/navigation.yaml'
    tuning = yaml.safe_load(path.read_text())['competition_reference']['ros__parameters']
    phase = LaunchConfiguration('phase').perform(context)
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    odometry = tuning['odometry'][phase].copy()
    odometry['base_frame'] = odometry.pop('base_footprint_frame')
    imu = odometry.pop('imu_topic')
    nodes = []
    wheel_topic = '/vel_raw'
    if phase == 'regions_1_3':
        center = tuning['feedback_center'].copy()
        wheel_input = center.pop('input_topic')
        command_input = center.pop('command_topic')
        center.pop('output_topic')
        wheel_topic = '/research/vel_raw_centered'
        nodes.append(Node(package='raicom_research_assets', executable='velocity_center',
                          parameters=[center, {'use_sim_time': sim_time}],
                          remappings=[('wheel_input', wheel_input), ('navigation_input', command_input),
                                      ('~/feedback', wheel_topic)]))
    nodes.append(Node(package='raicom_research_assets', executable='wheel_odometry',
                      parameters=[odometry, {'use_sim_time': sim_time}],
                      remappings=[('/vel_raw', wheel_topic), ('/imu0', imu)]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('phase', default_value='regions_1_3', choices=['regions_1_3', 'region_4']),
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        OpaqueFunction(function=start),
    ])
