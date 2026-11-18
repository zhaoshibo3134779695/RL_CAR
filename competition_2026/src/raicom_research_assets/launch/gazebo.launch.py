import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_path('raicom_research_assets')
    car = get_package_share_path('mowen')
    paths = [str(share.parent), str(car.parent), os.environ.get('GZ_SIM_RESOURCE_PATH', '')]
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=str(share / 'worlds/competition_map.sdf')),
        DeclareLaunchArgument('with_research_view', default_value='true', choices=['true', 'false']),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/research.launch.py')),
            condition=IfCondition(LaunchConfiguration('with_research_view')),
        ),
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', LaunchConfiguration('world')],
            additional_env={'GZ_SIM_RESOURCE_PATH': os.pathsep.join(p for p in paths if p)},
            output='screen', on_exit=Shutdown(),
        ),
    ])
