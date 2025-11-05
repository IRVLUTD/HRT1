#!/usr/bin/env python
#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2025).
#----------------------------------------------------------------------------------------------------
import sys
import rospy
sys.path.insert(0, "..")
from copy import deepcopy
from nav_msgs.msg import Path
from utils.ros_utils import rt_to_pose
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray



def publish_path(trajectory, parent_link="base_link", kill_at_end=False):
    """
    Method to publish a given trajectory as path of posestamped array
    Args:
        trajectory: list of 3d [x,y,z] or 6dof poses [x,y,z,x,y,z,w] or list of RT arrays or a combination of these as list
        parentlink: reference frame
        lifetime: time in secs to display the msg
    Output:
        3d path at /demonstration_path topic
    """

    path_msg = Path()
    path_msg.header.frame_id = parent_link

    pose_template = PoseStamped()
    pose_template.header.frame_id = parent_link

    for pose in trajectory:
        pose_template.header.stamp = rospy.Time.now()
        if isinstance(pose, list):
            pose_template.pose.position.x = pose[0]
            pose_template.pose.position.y = pose[1]
            pose_template.pose.position.z = pose[2]
        else:
            pose_template.pose.position.x = pose[:3, 3][0]
            pose_template.pose.position.y = pose[:3, 3][1]
            pose_template.pose.position.z = pose[:3, 3][2]
        path_msg.poses.append(deepcopy(pose_template))

    path_publisher = rospy.Publisher(
        "/trajectory_path", Path, queue_size=10)
    for _ in range(5):
        path_publisher.publish(path_msg)
        rospy.sleep(0.1)
    if kill_at_end:
        empty_msg = Path()
        empty_msg.header.frame_id = parent_link
        path_publisher.publish(empty_msg)


def create_gripper_viz_msg(
    trajectory,
    mesh_resource,
    parent_link="base_link",
    color=[1, 1, 1, 0.2],
    sampling_frequency=0.1,
    duration=5
):
    """
    visualizes gripper mesh at various points (subset) of the trajectory
    Args:
        trajectory: list of RT Matrices of trajectory
        mesh_resource: gripper mesh
        frame_id: default is "base_link"
        color: gripper mesh color [R,G,B,Alpha]
        sampling_frequency: full trajectory will be crowded. so display a uniform subset of it
        duration: for how long the visualization should last 
    returns MarkerArray message of gripper mesh set at various poses of the trajectory
    """
    gripper_viz_msg = MarkerArray()
    viz_marker = Marker()
    viz_marker.action = viz_marker.ADD
    viz_marker.header.frame_id = parent_link

    viz_marker.frame_locked = False
    viz_marker.lifetime = rospy.Duration(duration)

    viz_marker.type = viz_marker.MESH_RESOURCE
    viz_marker.mesh_resource = mesh_resource
    viz_marker.mesh_use_embedded_materials = True
    viz_marker.scale.x = 1
    viz_marker.scale.y = 1
    viz_marker.scale.z = 1.0

    for i, pose in enumerate(trajectory):
        if i == 0:
            color = [0, 0, 1, 0.6]
        else:
            # Calculate gradient color: red to yellow
            t = i / (len(trajectory) - 1)  # Normalized index [0, 1]
            # Interpolate from (1, 0, 0, 1) to (1, 1, 0, 1)
            color = [1.0, t, 0.0, 0.3]

        viz_marker.color.r = color[0]
        viz_marker.color.g = color[1]
        viz_marker.color.b = color[2]
        viz_marker.color.a = color[3]

        # if (i * sampling_frequency) % 1 == 0:
        viz_marker.id = i
        viz_marker.header.stamp = rospy.Time.now()
        viz_marker.pose = rt_to_pose(pose)
        gripper_viz_msg.markers.append(deepcopy(viz_marker))

    return gripper_viz_msg


def publish_trajectory(
    trajectory, parent_link="base_link", config=0, duration=10
):
    gripper_publisher = rospy.Publisher(
        "/gripper_viz/traj_viz", MarkerArray, queue_size=10)

    gripper_viz_msg = create_gripper_viz_msg(
        trajectory,
        mesh_resource=f"package://fetch_description/meshes/gripper_objs/{config}/open.obj",
        color=[0, 1, 0, 0.5],
        sampling_frequency=1,
        parent_link=parent_link,
        duration=duration
    )
    for _ in range(5):
        # Need to publish atleast 2 times, so rviz can actually capture the message before scripts/node die down.
        gripper_publisher.publish(gripper_viz_msg)
        rospy.sleep(0.1)


def publish_base_markers(delta_list, duration=10):

    # current positon is the base_link. so it will be [0,0,0]
    current_position = [0, 0 , 0]
    position_list = [current_position, delta_list]
    
    marker_array_publisher = rospy.Publisher(
        '/base_viz/marker_viz', MarkerArray, queue_size=10)
    marker_array = MarkerArray()
    colors = [[0, 1, 0], [0, 0, 1]]

    for i, position in enumerate(position_list):
        color = colors[i]
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "poses"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.b = color[0]
        marker.color.g = color[1]
        marker.color.r = color[2]
        marker.lifetime = rospy.Duration(duration)
        marker.frame_locked = False
        marker_array.markers.append(marker)

    for _ in range(5):
        marker_array_publisher.publish(marker_array)
        rospy.sleep(0.1)

    
