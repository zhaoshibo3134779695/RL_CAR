from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def playback_finished(event, context):
    if event.returncode:
        return [Shutdown(reason=f'Bag player exited with status {event.returncode}')]
    return []


def generate_launch_description():
    share = get_package_share_path('raicom_research_assets')
    trial = Path.cwd() / 'recordings/stage1_global_teb_compact_standard_ekf_20260810_trial7'
    description = (get_package_share_path('mowen') / 'urdf/mowen.urdf').read_text()
    topics = ['/move_base/local_costmap/costmap', '/move_base/TebLocalPlannerROS/local_plan',
              '/yellow_track_avoidance/corridor_goal', '/odom', '/odom_raw', '/yellow_track_avoidance/disc_scan', '/imu0']
    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value=str(trial)),
        DeclareLaunchArgument('domain_id', default_value='78'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain_id')),
        SetEnvironmentVariable('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST'),
        ExecuteProcess(cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag'), '--clock', '30',
                            '--delay', '2', '--disable-keyboard-controls', '--progress-bar-update-rate', '0', '--topics', *topics],
                       output='screen', on_exit=playback_finished),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'use_sim_time': True, 'robot_description': ParameterValue(description, value_type=str)}]),
        Node(package='raicom_research_assets', executable='recorded_odom_tf', parameters=[{'use_sim_time': True}]),
        Node(package='rviz2', executable='rviz2', name='trial_view',
             arguments=['-d', str(share / 'rviz/trial7.rviz')], parameters=[{'use_sim_time': True}],
             on_exit=Shutdown()),
    ])
