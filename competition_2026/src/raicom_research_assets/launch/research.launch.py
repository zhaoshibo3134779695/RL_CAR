from pathlib import Path
from xml.etree import ElementTree as ET

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def display_pose(context, share):
    pose = [0.0] * 6
    competition_map = Path(LaunchConfiguration('map').perform(context)).resolve() == (share / 'maps/vehicle/727best_map.yaml').resolve()
    if competition_map:
        world = ET.parse(share / 'worlds/competition_map.sdf')
        pose = [float(v) for v in world.findtext("./world/model[@name='mowen']/pose").split()]
    display_y = LaunchConfiguration('display_y').perform(context)
    reference = Node(package='raicom_research_assets', executable='competition_reference',
                     name='competition_reference',
                     parameters=[str(share / 'config/competition/navigation.yaml'),
                                 {'competition_map': competition_map}])
    if LaunchConfiguration('simulator_pose').perform(context) == 'true':
        return [reference]
    return [reference, Node(package='tf2_ros', executable='static_transform_publisher', name='display_pose',
                 arguments=['--x', str(pose[0]), '--y', display_y or str(pose[1]), '--yaw', str(pose[5]),
                            '--frame-id', 'map', '--child-frame-id', 'base_footprint'])]


def generate_launch_description():
    share = get_package_share_path('raicom_research_assets')
    description = (get_package_share_path('mowen') / 'urdf/mowen.urdf').read_text()
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=str(share / 'maps/vehicle/727best_map.yaml')),
        DeclareLaunchArgument('simulator_pose', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('display_y', default_value='', description='Display pose in the map frame, metres'),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / 'launch/map.launch.py')),
                                 launch_arguments={'map': LaunchConfiguration('map')}.items()),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': ParameterValue(description, value_type=str)}]),
        OpaqueFunction(function=display_pose, args=[share]),
        Node(package='raicom_research_assets', executable='track_perception', name='track_perception'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', str(share / 'rviz/research.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
