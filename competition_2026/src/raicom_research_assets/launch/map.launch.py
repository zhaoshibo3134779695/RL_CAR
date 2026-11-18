from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def start(context):
    share = get_package_share_path('raicom_research_assets')
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    names = ['map_server']
    actions = [Node(package='nav2_map_server', executable='map_server', name='map_server',
                    parameters=[{'yaml_filename': ParameterValue(LaunchConfiguration('map'), value_type=str),
                                 'use_sim_time': sim_time}])]
    if LaunchConfiguration('amcl').perform(context) == 'true':
        names.append('amcl')
        actions.append(Node(package='nav2_amcl', executable='amcl', name='amcl',
                            parameters=[str(share / 'config/replay/amcl.yaml'), {'use_sim_time': sim_time}]))
    actions.append(Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                        name='research_map_lifecycle',
                        parameters=[{'autostart': True, 'node_names': names, 'use_sim_time': sim_time}]))
    return actions


def generate_launch_description():
    share = get_package_share_path('raicom_research_assets')
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=str(share / 'maps/vehicle/727best_map.yaml')),
        DeclareLaunchArgument('amcl', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false']),
        OpaqueFunction(function=start),
    ])
