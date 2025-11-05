import rospy
import actionlib
from control_msgs.msg import PointHeadAction, PointHeadGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal


class FollowTrajectoryClient(object):
    def __init__(self, name, joint_names):
        self.client = actionlib.SimpleActionClient(
            "%s/follow_joint_trajectory" % name, FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for %s..." % name)
        self.client.wait_for_server()
        self.joint_names = joint_names

    def move_to(self, positions, duration=5.0):
        if len(self.joint_names) != len(positions):
            print("Invalid trajectory position")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = positions
        trajectory.points[0].velocities = [0.0 for _ in positions]
        trajectory.points[0].accelerations = [0.0 for _ in positions]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory

        self.client.send_goal(follow_goal)
        self.client.wait_for_result()


class PointHeadClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            "head_controller/point_head", PointHeadAction
        )
        rospy.loginfo("Waiting for head_controller...")
        self.success = self.client.wait_for_server(timeout=rospy.Duration(3.0))
        if (self.success is False):
            rospy.loginfo("no point head controller available")
        else:
            rospy.loginfo("Use head_controller/point_head")

    def look_at(self, x, y, z, frame="base_link", duration=1.0):
        """
        Turning head to look at x,y,z
        x: x location
        y: y location
        z: z location
        frame: reference frame
        duration: movement time
        :return:
        """
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()


if __name__ == "__main__":
    rospy.init_node("test_controls")
    tcontrol = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    tcontrol.move_to([0])
    
    hcontrol = PointHeadClient()
    hcontrol.look_at(1, 0.2, 0.8, "base_link")

