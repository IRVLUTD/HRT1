#----------------------------------------------------------------------------------------------------
# Trial Data Recorder for TTO Execution
# Records synchronized joint states, gripper state, base position, and images
# directly via ROS subscribers and saves them to an HDF5 file.
#----------------------------------------------------------------------------------------------------
import os
import tf
import h5py
import rospy
import tf2_ros
import threading
import ros_numpy
import numpy as np
import message_filters
from copy import deepcopy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo

# Quaternion to rotation matrix (same convention as ros_utils)
from transforms3d.quaternions import quat2mat


def _ros_qt_to_rt(rot, trans):
    """Convert ROS quaternion + translation to 4x4 RT matrix."""
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]  # w
    qt[1] = rot[0]  # x
    qt[2] = rot[1]  # y
    qt[3] = rot[2]  # z
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans
    return obj_T


def _odometry_to_rt(odometry):
    """Convert Odometry message to 4x4 RT matrix."""
    trans = [
        odometry.pose.pose.position.x,
        odometry.pose.pose.position.y,
        odometry.pose.pose.position.z,
    ]
    quat = odometry.pose.pose.orientation
    rot = [quat.x, quat.y, quat.z, quat.w]
    return _ros_qt_to_rt(rot, trans)


# Gripper thresholds
GRIPPER_CLOSED_THRESHOLD = 0.01
GRIPPER_OPEN_THRESHOLD   = 0.04

# Arm joint names for Fetch (7 DOF)
FETCH_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]
GRIPPER_JOINT_NAME = "l_gripper_finger_joint"


class TrialDataRecorder:
    """
    Records synchronized robot data (images, joint states, gripper, base pose)
    using ROS ApproximateTimeSynchronizer and saves to HDF5.

    Subscribes directly to:
      - /head_camera/rgb/image_raw  (Image)
      - /odom                       (Odometry)
      - /joint_states               (JointState)

    All three are synced via ApproximateTimeSynchronizer so each recorded
    datapoint has image, joints, and base pose from the same moment.

    Usage:
        recorder = TrialDataRecorder()
        recorder.save_initial_data(trajectory, base_pose)
        recorder.start_recording(phase="base_move")
        ...
        recorder.pause_recording()
        ...
        recorder.start_recording(phase="arm_execute")
        ...
        recorder.record_single_datapoint(phase="arm_execute")
        recorder.stop_recording()
        recorder.save_to_hdf5(filepath, trajectory_name, language_instruction)
    """

    def __init__(self, motionplanner, record_interval=0.1,
                 arm_joint_names=None, gripper_joint_name=None):
        """
        Args:
            motionplanner: MotionPlanning instance (for joint states via get_robot_state())
            record_interval: minimum time between recorded datapoints in seconds (~10 Hz)
            arm_joint_names: list of 7 arm joint names (default: Fetch arm joints)
            gripper_joint_name: gripper joint name (default: l_gripper_finger_joint)
        """
        self.record_interval = record_interval
        self.motionplanner = motionplanner
        self.arm_joint_names = arm_joint_names or FETCH_ARM_JOINT_NAMES
        self.gripper_joint_name = gripper_joint_name or GRIPPER_JOINT_NAME

        # Data buffers
        self.timestamps = []
        self.images_fetch = []
        self.images_realsense = []
        self.joint_states = []       # (T, 7) arm joints
        self.gripper_states = []     # (T,) raw gripper position
        self.gripper_labels = []     # (T,) "open" / "closed" / "partial"
        self.base_positions = []     # (T, 4, 4)
        self.phases = []             # "base_move" or "arm_execute"

        # Initial data (saved once)
        self.ref_trajectory = None
        self.initial_base_pose = None

        # Recording control
        self._is_recording = False
        self._current_phase = "base_move"
        self._lock = threading.Lock()
        self._last_recorded_time = None

        # Latest synced snapshot (always updated by callback regardless of recording state)
        self._latest_rgb = None
        self._latest_joint_positions = None
        self._latest_joint_names = None
        self._latest_base_pose = None
        self._latest_stamp = None
        self._snapshot_lock = threading.Lock()

        # ---- ROS Subscribers with ApproximateTimeSynchronizer ----
        rgb_sub = message_filters.Subscriber(
            "/head_camera/rgb/image_raw", Image, queue_size=10
        )
        wrist_sub = message_filters.Subscriber(
            "/realsense/camera/image", Image, queue_size=10
        )
        odom_sub = message_filters.Subscriber(
            "/odom", Odometry, queue_size=10
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, wrist_sub, odom_sub],
            queue_size=5,
            slop=0.5,
        )
        ts.registerCallback(self._synced_callback)
        # keep reference so it doesn't get garbage collected
        self._ts = ts
        self._subs = [rgb_sub, wrist_sub, odom_sub]

        rospy.loginfo("[TrialDataRecorder] Initialized with synced subscribers "
                      "(image_fetch + image_realsense + odom) + MoveIt joint states")

    # ------------------------------------------------------------------ #
    #  Synced ROS Callback
    # ------------------------------------------------------------------ #

    def _synced_callback(self, rgb_msg, wrist_msg, odom_msg):
        """
        Called by ApproximateTimeSynchronizer when all three messages
        arrive approximately at the same time. Joint states are queried
        from MoveIt at this moment.
        """
        # Convert RGB images
        rgb_cv = ros_numpy.numpify(rgb_msg)
        wrist_cv = ros_numpy.numpify(wrist_msg)

        # Convert odom → 4x4
        base_pose = _odometry_to_rt(odom_msg)

        # Get joint states from MoveIt (same as original run.py)
        try:
            robot_state = self.motionplanner.get_robot_state()
            joint_positions = list(robot_state.joint_state.position)
            joint_names = list(robot_state.joint_state.name)
        except Exception as e:
            rospy.logwarn_throttle(2, f"[TrialDataRecorder] Failed to get robot state: {e}")
            return

        stamp = rgb_msg.header.stamp.to_sec()

        # Always update latest snapshot
        with self._snapshot_lock:
            self._latest_rgb = rgb_cv.copy()
            self._latest_wrist = wrist_cv.copy()
            self._latest_base_pose = base_pose.copy()
            self._latest_joint_positions = joint_positions
            self._latest_joint_names = joint_names
            self._latest_stamp = stamp

        # If recording, append to buffers (throttled by record_interval)
        with self._lock:
            if not self._is_recording:
                return
            current_time = rospy.get_time()
            if (self._last_recorded_time is not None and
                    (current_time - self._last_recorded_time) < self.record_interval):
                return
            self._last_recorded_time = current_time

        # Extract arm joint angles (7 DOF)
        arm_joints = np.array([
            joint_positions[joint_names.index(name)]
            for name in self.arm_joint_names
        ], dtype=np.float32)

        # Extract gripper state + derive label
        gripper_pos = float(joint_positions[joint_names.index(self.gripper_joint_name)])
        if gripper_pos < GRIPPER_CLOSED_THRESHOLD:
            gripper_label = "closed"
        elif gripper_pos > GRIPPER_OPEN_THRESHOLD:
            gripper_label = "open"
        else:
            gripper_label = "partial"

        self.timestamps.append(stamp)
        self.images_fetch.append(rgb_cv.copy())
        self.images_realsense.append(wrist_cv.copy())
        self.joint_states.append(arm_joints)
        self.gripper_states.append(gripper_pos)
        self.gripper_labels.append(gripper_label)
        self.base_positions.append(base_pose.copy())
        self.phases.append(self._current_phase)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def save_initial_data(self, ref_trajectory, initial_base_pose):
        """Save the reference trajectory and initial base pose."""
        self.ref_trajectory = np.array(ref_trajectory)
        self.initial_base_pose = np.array(initial_base_pose)
        rospy.loginfo("[TrialDataRecorder] Initial data saved (trajectory + base pose)")

    def start_recording(self, phase="base_move"):
        """Start or resume recording. Data will be appended in the synced callback."""
        with self._lock:
            if self._is_recording:
                rospy.logwarn("[TrialDataRecorder] Already recording, ignoring start call")
                return
            self._current_phase = phase
            self._is_recording = True
            self._last_recorded_time = None  # record first datapoint immediately
        rospy.loginfo(f"[TrialDataRecorder] Recording started (phase={phase})")

    def pause_recording(self):
        """Pause recording. Data is retained, can resume with start_recording."""
        with self._lock:
            self._is_recording = False
            self._last_recorded_time = None
        rospy.loginfo(f"[TrialDataRecorder] Recording paused "
                      f"({len(self.timestamps)} datapoints so far)")

    def stop_recording(self):
        """Stop recording entirely."""
        self.pause_recording()
        rospy.loginfo(f"[TrialDataRecorder] Recording stopped. "
                      f"Total datapoints: {len(self.timestamps)}")

    def record_single_datapoint(self, phase=None):
        """Record a single synchronized datapoint from the latest snapshot."""
        with self._snapshot_lock:
            if self._latest_rgb is None:
                rospy.logwarn("[TrialDataRecorder] No data available yet for single datapoint")
                return
            rgb = self._latest_rgb.copy()
            wrist = self._latest_wrist.copy()
            base_pose = self._latest_base_pose.copy()
            joint_positions = list(self._latest_joint_positions)
            joint_names = list(self._latest_joint_names)
            stamp = self._latest_stamp

        p = phase if phase is not None else self._current_phase

        arm_joints = np.array([
            joint_positions[joint_names.index(name)]
            for name in self.arm_joint_names
        ], dtype=np.float32)

        gripper_pos = float(joint_positions[joint_names.index(self.gripper_joint_name)])
        if gripper_pos < GRIPPER_CLOSED_THRESHOLD:
            gripper_label = "closed"
        elif gripper_pos > GRIPPER_OPEN_THRESHOLD:
            gripper_label = "open"
        else:
            gripper_label = "partial"

        self.timestamps.append(stamp)
        self.images_fetch.append(rgb)
        self.images_realsense.append(wrist)
        self.joint_states.append(arm_joints)
        self.gripper_states.append(gripper_pos)
        self.gripper_labels.append(gripper_label)
        self.base_positions.append(base_pose)
        self.phases.append(p)

        rospy.loginfo(f"[TrialDataRecorder] Recorded single datapoint "
                      f"(gripper={gripper_label}, phase={p})")

    # ------------------------------------------------------------------ #
    #  HDF5 Save
    # ------------------------------------------------------------------ #

    def save_to_hdf5(self, filepath, trajectory_name, language_instruction=""):
        """
        Write all buffered data to an HDF5 file.

        Args:
            filepath: path to the .h5 file to create
            trajectory_name: name for the trajectory group
            language_instruction: text description of the task
        """
        T = len(self.timestamps)
        if T == 0:
            rospy.logerr("[TrialDataRecorder] No data recorded, skipping HDF5 save")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            grp = f.create_group(f"data/{trajectory_name}")

            # Language instruction
            grp.create_dataset("language_instruction", data=language_instruction)

            # Reference trajectory (N, 4, 4)
            if self.ref_trajectory is not None:
                grp.create_dataset("ref_trajectory",
                                   data=self.ref_trajectory.astype(np.float32))

            # Initial base pose (4, 4)
            if self.initial_base_pose is not None:
                grp.create_dataset("initial_base_pose",
                                   data=self.initial_base_pose.astype(np.float32))

            # Observations group
            obs = grp.create_group("obs")

            # Timestamps
            obs.create_dataset("timestamp",
                             data=np.array(self.timestamps, dtype=np.float64))

            # Frame indices
            obs.create_dataset("frame_index",
                             data=np.arange(T, dtype=np.int32))

            # Images (T, H, W, 3) — gzip compressed, chunked per frame
            images_array = np.stack(self.images_fetch, axis=0).astype(np.uint8)
            obs.create_dataset("image_fetch", data=images_array,
                             chunks=(1, images_array.shape[1], images_array.shape[2], 3),
                             compression="gzip", compression_opts=4)

            # Wrist camera images (T, H, W, 3)
            wrist_array = np.stack(self.images_realsense, axis=0).astype(np.uint8)
            obs.create_dataset("image_realsense", data=wrist_array,
                             chunks=(1, wrist_array.shape[1], wrist_array.shape[2], 3),
                             compression="gzip", compression_opts=4)

            # Joint states (T, 7) — 7 DOF arm
            obs.create_dataset("state",
                             data=np.stack(self.joint_states, axis=0))

            # Gripper — raw position (T,)
            obs.create_dataset("gripper_state",
                             data=np.array(self.gripper_states, dtype=np.float32))

            # Gripper — derived label (T,)
            obs.create_dataset("gripper_label",
                             data=np.array(self.gripper_labels, dtype=h5py.string_dtype()))

            # Base position (T, 4, 4)
            obs.create_dataset("base_position",
                             data=np.stack(self.base_positions, axis=0).astype(np.float32))

            # Action group — phase labels
            action = grp.create_group("action")
            action.create_dataset("phase",
                                data=np.array(self.phases, dtype=h5py.string_dtype()))

        rospy.loginfo(f"[TrialDataRecorder] Saved {T} datapoints to {filepath}")
        rospy.loginfo(f"  - image_fetch shape: {images_array.shape}")
        rospy.loginfo(f"  - image_realsense shape: {wrist_array.shape}")
        rospy.loginfo(f"  - state shape: {np.stack(self.joint_states, axis=0).shape}")
        rospy.loginfo(f"  - gripper labels: { {l: self.gripper_labels.count(l) for l in set(self.gripper_labels)} }")
        rospy.loginfo(f"  - phases: { {p: self.phases.count(p) for p in set(self.phases)} }")
