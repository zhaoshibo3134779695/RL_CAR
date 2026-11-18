from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def start(context, share):
    map_file = LaunchConfiguration('map')
    config = str(share / 'config/replay/planner.yaml')
    managed = ['planner_server']
    motion = LaunchConfiguration('motion').perform(context) == 'true'
    interactive = LaunchConfiguration('interactive').perform(context) == 'true'
    competition_map = Path(map_file.perform(context)).resolve() == (share / 'maps/vehicle/727best_map.yaml').resolve()
    if motion and not (interactive and competition_map):
        raise ValueError('The motion world requires interactive:=true and the 727best competition map.')
    if interactive:
        actions = [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/research.launch.py')),
            launch_arguments={'map': map_file, 'simulator_pose': str(motion).lower()}.items())]
        if competition_map:
            world = 'competition_motion.sdf' if motion else 'competition_map.sdf'
            if motion:
                actions.append(Node(package='raicom_research_assets', executable='simulate_route', on_exit=Shutdown()))
            actions.append(Node(package='raicom_research_assets', executable='recorded_waypoints'))
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(share / 'launch/gazebo.launch.py')),
                launch_arguments={'with_research_view': 'false', 'world': str(share / 'worlds' / world)}.items()))
        config = RewrittenYaml(source_file=config, param_rewrites={'robot_base_frame': 'base_footprint'})
        actions.append(Node(package='raicom_research_assets', executable='rviz_plan_goal'))
    else:
        actions = [
            Node(package='tf2_ros', executable='static_transform_publisher', name='planning_origin',
                 arguments=['--frame-id', 'map', '--child-frame-id', 'research_planning_origin']),
            Node(package='nav2_map_server', executable='map_server', name='map_server',
                 parameters=[{'yaml_filename': ParameterValue(map_file, value_type=str)}]),
        ]
        managed.insert(0, 'map_server')
    actions.extend([
        Node(package='nav2_planner', executable='planner_server', name='planner_server', parameters=[config]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='planning_lifecycle',
             parameters=[{'autostart': True, 'node_names': managed}]),
    ])
    return actions


def generate_launch_description():
    share = get_package_share_path('raicom_research_assets')
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=str(share / 'maps/vehicle/727best_map.yaml')),
        DeclareLaunchArgument('motion', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('interactive', default_value='false', choices=['true', 'false']),
        OpaqueFunction(function=start, args=[share]),
    ])
