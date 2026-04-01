#!/usr/bin/env python
#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2026).
#----------------------------------------------------------------------------------------------------
"""
Evaluation pipeline for deploying a learned policy on the Fetch robot.

The policy outputs a chunk of actions at each query, shape (chunk_size, 8):
    Each row: [7 joint angle deltas, 1 gripper value]
    e.g. chunk_size=8 means the policy predicts the next 8 timesteps.

Execution loop:
    1. Read current joint angles + observations
    2. Query policy → get (chunk_size, 8) action chunk
    3. For each timestep in the chunk:
        a. Read current joint angles
        b. Compute target = current + delta
        c. Clip target to joint limits
        d. Execute arm movement
        e. Execute gripper open/close
    4. Repeat from step 1
"""
import sys
sys.path.insert(0, "..")
import os
import time
import argparse
import rospy
import numpy as np
from abc import ABC, abstractmethod

from utils.listener import Listener
from utils.control_utils import PointHeadClient, FollowTrajectoryClient
from utils.moveit_utils import MotionPlanning, GripperPlanning


# --------------------------------------------------------------------------- #
#  Fetch Arm Joint Limits (from URDF)
# --------------------------------------------------------------------------- #
FETCH_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# (lower, upper) in radians
FETCH_ARM_JOINT_LIMITS = np.array([
    [-1.6056, 1.6056],   # shoulder_pan
    [-1.221,  1.518],    # shoulder_lift
    [-3.14159, 3.14159], # upperarm_roll  (continuous → use large range)
    [-2.251,  2.251],    # elbow_flex
    [-3.14159, 3.14159], # forearm_roll   (continuous)
    [-2.16,   2.16],     # wrist_flex
    [-3.14159, 3.14159], # wrist_roll     (continuous)
])

JOINT_LOWER = FETCH_ARM_JOINT_LIMITS[:, 0]
JOINT_UPPER = FETCH_ARM_JOINT_LIMITS[:, 1]


# --------------------------------------------------------------------------- #
#  Policy Interface
# --------------------------------------------------------------------------- #
class PolicyInterface(ABC):
    """
    Base class for policies. Subclass this and implement `predict`.
    """

    @abstractmethod
    def predict(self, observation: dict) -> np.ndarray:
        """
        Given an observation dict, return an action chunk of shape (chunk_size, 8).
            action[:, 0:7]  — joint angle deltas for the 7-DOF arm
            action[:, 7]    — gripper value (0 = close, 1 = open)

        chunk_size is the number of future timesteps the policy predicts.
        """
        raise NotImplementedError


class DummyPolicy(PolicyInterface):
    """
    No-op policy that returns zeros for 8 timesteps.
    Useful for testing the pipeline — the robot should stay in place.
    """
    def __init__(self, chunk_size: int = 8):
        self.chunk_size = chunk_size

    def predict(self, observation: dict) -> np.ndarray:
        return np.zeros((self.chunk_size, 8))


# --------------------------------------------------------------------------- #
#  Evaluation Pipeline
# --------------------------------------------------------------------------- #
class EvaluationPipeline:
    """
    Runs a policy in a closed loop on the Fetch robot.

    Args:
        policy:            PolicyInterface instance
        max_steps:         maximum number of control steps
        gripper_threshold: boundary for open (>= threshold) vs close
        use_listener:      whether to initialise the camera Listener
                           (set True if the policy needs images)
        step_duration:     time (s) allocated for each joint move
    """

    def __init__(
        self,
        policy: PolicyInterface,
        max_steps: int = 100,
        gripper_threshold: float = 0.5,
        use_listener: bool = False,
        step_duration: float = 0.5,
    ):
        self.policy = policy
        self.max_steps = max_steps
        self.gripper_threshold = gripper_threshold
        self.step_duration = step_duration

        # ---- ROS controllers ---- #
        rospy.loginfo("Initialising MotionPlanning …")
        self.motionplanner = MotionPlanning()

        rospy.loginfo("Initialising GripperPlanning …")
        self.gripperplanner = GripperPlanning()

        rospy.loginfo("Initialising head and torso controllers …")
        self.head_action = PointHeadClient()
        self.torso_action = FollowTrajectoryClient(
            "torso_controller", ["torso_lift_joint"]
        )

        # ---- optional camera listener ---- #
        self.listener = None
        if use_listener:
            rospy.loginfo("Initialising Listener (camera / odom) …")
            self.listener = Listener()

        # gripper state tracking to avoid redundant commands
        self.gripper_is_open = True  # assume starts open

        rospy.loginfo("Evaluation pipeline ready.")

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #
    def get_observation(self) -> dict:
        """
        Build the observation dict that is passed to the policy.
        Contains at minimum the current joint angles.
        If a Listener is available, also includes rgb, depth, etc.
        """
        joint_angles = np.array(
            self.motionplanner.get_active_joint_angles(), dtype=np.float64
        )

        obs = {"joint_angles": joint_angles}

        if self.listener is not None:
            rgb, depth, RT_camera, base_pose = self.listener.get_data()
            obs["rgb"] = rgb
            obs["depth"] = depth
            obs["RT_camera"] = RT_camera
            obs["base_pose"] = base_pose

        return obs

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #
    def compute_target_joints(
        self, current_joints: np.ndarray, delta_joints: np.ndarray
    ) -> np.ndarray:
        """
        target = current + delta, clipped to joint limits.
        """
        # target = current_joints + delta_joints
        target = delta_joints
        target = np.clip(target, JOINT_LOWER, JOINT_UPPER)
        return target

    def execute_arm(self, target_joints: np.ndarray):
        """
        Move the arm to target_joints using MoveIt.
        """
        self.motionplanner.move_to_joint_angle(target_joints.tolist())

    def execute_gripper(self, gripper_value: float):
        """
        Open or close the gripper based on the threshold.
        Skips redundant commands.
        """
        should_open = gripper_value >= self.gripper_threshold

        if should_open and not self.gripper_is_open:
            rospy.loginfo("Gripper → OPEN")
            self.gripperplanner.open()
            self.gripper_is_open = True
        elif not should_open and self.gripper_is_open:
            rospy.loginfo("Gripper → CLOSE")
            self.gripperplanner.close()
            self.gripper_is_open = False

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        """
        Execute the evaluation loop.

        Outer loop: queries policy once per iteration → gets action chunk.
        Inner loop: executes each timestep in the chunk sequentially.
        Total executed steps are tracked against max_steps.
        """
        rospy.loginfo(
            f"Starting evaluation: max_steps={self.max_steps}, "
            f"gripper_threshold={self.gripper_threshold}"
        )

        total_steps = 0

        while total_steps < self.max_steps:
            if rospy.is_shutdown():
                rospy.logwarn("ROS shutdown detected — stopping evaluation.")
                break

            # 1. Observe (once per chunk)
            obs = self.get_observation()

            # 2. Query policy → (chunk_size, 8)
            action_chunk = self.policy.predict(obs)
            assert action_chunk.ndim == 2 and action_chunk.shape[1] == 8, (
                f"Policy must return shape (chunk_size, 8), got {action_chunk.shape}"
            )
            chunk_size = action_chunk.shape[0]
            rospy.loginfo(
                f"Policy returned action chunk of size {chunk_size}"
            )

            # 3. Execute each timestep in the chunk
            for t in range(chunk_size):
                if rospy.is_shutdown() or total_steps >= self.max_steps:
                    break

                # Re-read current joints before each sub-step
                current_joints = np.array(
                    self.motionplanner.get_active_joint_angles(), dtype=np.float64
                )

                delta_joints = action_chunk[t, :7]
                gripper_value = action_chunk[t, 7]

                # Compute target
                target_joints = self.compute_target_joints(
                    current_joints, delta_joints
                )

                total_steps += 1
                rospy.loginfo(
                    f"[Step {total_steps}/{self.max_steps}]  "
                    f"chunk_t={t+1}/{chunk_size}  "
                    f"Δnorm={np.linalg.norm(delta_joints):.4f}  "
                    f"gripper={gripper_value:.2f}"
                )

                # Execute
                self.execute_arm(target_joints)
                self.execute_gripper(gripper_value)

        rospy.loginfo(f"Evaluation complete. Total steps executed: {total_steps}")
        self.motionplanner.stop()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on the Fetch robot."
    )
    parser.add_argument(
        "--max_steps", type=int, default=100,
        help="Maximum number of control steps",
    )
    parser.add_argument(
        "--gripper_threshold", type=float, default=0.5,
        help="Gripper value >= this → open, < this → close",
    )
    parser.add_argument(
        "--step_duration", type=float, default=0.5,
        help="Duration (s) per arm movement step",
    )
    parser.add_argument(
        "--use_listener", action="store_true",
        help="Initialise camera / odom Listener for image observations",
    )
    args = parser.parse_args()

    rospy.init_node("evaluate_policy")
    rospy.sleep(1)

    # ---- Plug in your policy here ---- #
    policy = DummyPolicy(chunk_size=8)
    # e.g.  policy = MyACTPolicy("checkpoints/act_best.pth")

    pipeline = EvaluationPipeline(
        policy=policy,
        max_steps=args.max_steps,
        gripper_threshold=args.gripper_threshold,
        use_listener=args.use_listener,
        step_duration=args.step_duration,
    )
    pipeline.run()
