import time
import rospy
import numpy as np
from geometry_msgs.msg import Pose
from transforms3d.quaternions import mat2quat, quat2mat
import tf.transformations as tf
from transforms3d.quaternions import quat2mat
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo


def convert_rosqt_to_standard(pose_ros):
    """
    Args:
        pose_ros - list of [position, quaternion] 1x7
    Converts (posn, x,y,z,w) quat to (posn, w,x,y,z) quat"""
    posn = pose_ros[:3]
    ros_qt = pose_ros[3:]
    quat = [ros_qt[-1], ros_qt[0], ros_qt[1], ros_qt[2]]
    return [*posn, *quat]


def convert_standard_to_rosqt(pose_s):
    """Converts (posn, w,x,y,z) quat to ROS format (posn, x,y,z,w) quat"""
    posn = pose_s[:3]
    q_s = pose_s[3:]
    quat = [q_s[1], q_s[2], q_s[3], q_s[0]]
    return [*posn, *quat]


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def ros_pose_to_rt(pose):
    qarray = [0, 0, 0, 0]
    qarray[0] = pose.orientation.x
    qarray[1] = pose.orientation.y
    qarray[2] = pose.orientation.z
    qarray[3] = pose.orientation.w

    t = [0, 0, 0]
    t[0] = pose.position.x
    t[1] = pose.position.y
    t[2] = pose.position.z

    return ros_qt_to_rt(qarray, t)


def rt_to_ros_pose(pose, rt):
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]

    return pose


def rt_to_ros_qt(rt):
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    return quat, trans


def get_relative_pose_from_tf(listener, source_frame, target_frame):
    first_time = True
    time_start = time.time()
    while time.time() - time_start < 3:
        try:
            init_trans, init_rot = listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            break
        except Exception as e:
            if first_time:
                print(str(e))
            init_trans = np.array([0, 0, 0])
            init_rot = np.array([0, 0, 0, 1])
            continue

    # print('got relative pose between {} and {}'.format(source_frame, target_frame))
    return ros_qt_to_rt(init_rot, init_trans)


def rt_to_pose(rt_matrix):
    """Convert a 4x4 RT matrix to a ROS Pose message."""
    pose = Pose()
    position = rt_matrix[:3, 3]  # Extract translation
    pose.position.x, pose.position.y, pose.position.z = position

    # Extract quaternion from rotation matrix
    quaternion = tf.quaternion_from_matrix(rt_matrix)
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]
    return pose


def xyz_euler_to_pose(position, angles):
    quat = tf.quaternion_from_euler(*angles)
    rt = ros_qt_to_rt(quat, position)
    pose = rt_to_pose(rt)
    return pose


def odometry_to_rt(odometry):
    """
    converts odometry msg into Rt matrix
    """
    trans = [
        odometry.pose.pose.position.x,
        odometry.pose.pose.position.y,
        odometry.pose.pose.position.z,
    ]
    quat = odometry.pose.pose.orientation
    rot = [quat.x, quat.y, quat.z, quat.w]

    return ros_qt_to_rt(rot, trans)


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def unpack_pose(pose, rot_first=False):
    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = quat2mat(pose[:4])
        unpacked[:3, 3] = pose[4:]
    else:
        unpacked[:3, :3] = quat2mat(pose[3:])
        unpacked[:3, 3] = pose[:3]
    return unpacked


def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def convert_plan_to_trajectory_toppra(robot, joint_names, plan, is_show=False):

    ndof = plan.shape[0]
    T = plan.shape[1]
    ss = np.linspace(0, 1, T)
    way_pts = plan.T
    vlims = robot.velocity_optimized_joint_limits.toarray().flatten() * 0.2
    alims = np.ones(ndof) * 0.1

    path = ta.SplineInterpolator(ss, way_pts, bc_type="natural")
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(alims)

    instance = algo.TOPPRA([pc_vel, pc_acc], path,
                           parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()

    num1 = 50
    num2 = 50
    num = num1 + num2
    # num=len(way_pts)
    total_exec_time = jnt_traj.duration
    ts_sample1 = np.linspace(0, total_exec_time / 2, num1)
    ts_sample2 = np.linspace(total_exec_time / 2, total_exec_time, num2)
    ts_sample = np.concatenate((ts_sample1, ts_sample2))

    ts_sample = np.linspace(0, total_exec_time, num)

    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)
    if is_show:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, sharex=True)
        for i in range(path.dof):
            # plot the i-th joint trajectory
            axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
            axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
            axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        plt.show()

    trajectory = JointTrajectory()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.joint_names = joint_names

    for i in range(num):
        point = JointTrajectoryPoint()
        point.positions = qs_sample[i]
        point.velocities = qds_sample[i]
        point.accelerations = qdds_sample[i]
        point.time_from_start = rospy.Duration.from_sec(ts_sample[i])
        trajectory.points.append(point)
    return trajectory
