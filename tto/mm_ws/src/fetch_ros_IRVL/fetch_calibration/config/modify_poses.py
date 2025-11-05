import rosbag
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import rospy
import math
# Open the bag file
bag = rosbag.Bag('/home/felipe/ft_ros/src/fetch_ros_IRVL/fetch_calibration/config/calibration_poses.bag')
out_bag = rosbag.Bag('/home/felipe/ft_ros/src/fetch_ros_IRVL/fetch_calibration/config/calibration_poses_modified.bag','w')

# Read messages from the bag file for specific topics
for topic, msg, t in bag.read_messages():
    #print(f"Topic: {topic},{msg}, Timestamp: {t}")
    joint_names = [
    "l_gripper_finger_joint", "r_gripper_finger_joint", "l_wheel_joint", 
    "r_wheel_joint", "torso_lift_joint", "bellows_joint", "head_pan_joint",
    "head_tilt_joint", "shoulder_pan_joint", "shoulder_lift_joint", 
    "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", 
    "wrist_flex_joint", "wrist_roll_joint"
    ]


    joint_positions = list(msg.position)
    velocity = list(msg.velocity)
    effort = list(msg.effort)
    joint_state = JointState()
    joint_state.header = Header()
    joint_state.header.seq = 0
    joint_state.header.stamp.secs = 0
    joint_state.header.stamp.nsecs = 0
    joint_state.header.frame_id = ''
    joint_state.name = joint_names
    wrist_roll_index = joint_names.index('wrist_roll_joint')
    torso_lift_index = joint_names.index('torso_lift_joint')
    joint_positions[wrist_roll_index] += 0.8287
    if joint_positions[wrist_roll_index] > 3.05:
        joint_positions[wrist_roll_index] -= 2* math.pi
        #print("ERROR - Wrist Roll Joint out of range")
        #print(joint_positions[wrist_roll_index])
    elif joint_positions[wrist_roll_index] < -3.05:
        joint_positions[wrist_roll_index] += 2* math.pi
        #print("ERROR - Wrist Roll Joint out of range")
        #print(joint_positions[wrist_roll_index])

    # Modify 'wrist_roll_joint' position
    for name in joint_names:
        if name in ['upperarm_roll_joint', "forearm_roll_joint", 'wrist_roll_joint']:
            index = joint_names.index(name)
            if joint_positions[index]> 3.05:
                joint_positions[index] = 3.05
                print(name + " is out of range")
                #print(joint_positions[index])
            elif joint_positions[index] < -3.05:
                joint_positions[index] = -3.05
                print(name + " is out of range")
                #print(joint_positions[index])

    
    

    # Modify 'torso_lift_joint' position
    print(joint_positions[torso_lift_index])
    joint_positions[torso_lift_index] = 0.35
    
    # Write the modified message to the new bag file
    joint_state.position = joint_positions
    joint_state.velocity = []  # Empty list for velocity
    joint_state.effort = []    # Empty list for effort


    out_bag.write('calibration_joint_states', joint_state, t=t)

    

# Close the bag file
bag.close()
out_bag.close()