#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2025).
#----------------------------------------------------------------------------------------------------
import os, sys
sys.path.insert(0,"..")
import rospy
from gazebo_msgs.msg import ModelState
from transforms3d.euler import quat2euler 
from gazebo_msgs.srv import GetModelState, SetModelState, SpawnModel, DeleteModel
from utils.ros_utils import ros_pose_to_rt
from geometry_msgs.msg import Pose

class objectService:
    def __init__(self, get_service_name='/gazebo/get_model_state', set_service_name='/gazebo/set_model_state',spawn_service_name='/gazebo/spawn_sdf_model'):
        rospy.wait_for_service(get_service_name)
        self.obj_get_service = rospy.ServiceProxy(get_service_name, GetModelState)

        rospy.wait_for_service(set_service_name)
        self.obj_set_service = rospy.ServiceProxy(set_service_name, SetModelState)

        rospy.wait_for_service(spawn_service_name)
        self.model_spawn_service = rospy.ServiceProxy(spawn_service_name, SpawnModel)

        rospy.wait_for_service('/gazebo/delete_model')
        self.model_delete_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.new_state = ModelState()

        self.object_pose_inq = [0,0,0,0,0,0,0]
        self.object_pose_eul = [0,0,0,0,0,0]
        self.reference = None

    def convert_quat_to_euler(self):
        euler_orientation = quat2euler(self.object_pose_inq[3:])
        self.object_pose_eul[0:3] = self.object_pose_inq[0:3]
        self.object_pose_eul[3:] =  euler_orientation

    def get_state(self, obj_name):
        object_pose = self.obj_get_service(obj_name,'').pose
        self.object_pose_inq[0] = object_pose.position.x
        self.object_pose_inq[1] = object_pose.position.y
        self.object_pose_inq[2] = object_pose.position.z
        self.object_pose_inq[3] = object_pose.orientation.x
        self.object_pose_inq[4] = object_pose.orientation.y
        self.object_pose_inq[5] = object_pose.orientation.z
        self.object_pose_inq[6] = object_pose.orientation.w
        # self.convert_quat_to_euler()
        return self.object_pose_inq

    def set_state(self, obj_name, target_pose=[0,0,0.75,0,0,0,1]):
        self.new_state.model_name = obj_name
        self.new_state.pose.position.x = target_pose[0]
        self.new_state.pose.position.y = target_pose[1]
        self.new_state.pose.position.z = target_pose[2]
        self.new_state.pose.orientation.x = target_pose[3]
        self.new_state.pose.orientation.y = target_pose[4]
        self.new_state.pose.orientation.z = target_pose[5]
        self.new_state.pose.orientation.w = target_pose[6]
        response = self.obj_set_service(self.new_state)
        if response.success==True:
            print(f"{obj_name}: state set successfully")
        else:
            print(f"{obj_name}: failed to set state! object doesn't exists or invalid pose")

    def get_object_RT(self, obj_name):
        object_pose = self.obj_get_service(obj_name,'').pose
        self.obj_pose_RT = ros_pose_to_rt(object_pose)
        return self.obj_pose_RT

    def spawn_model(self, model_name, target_pose=[0,0,0.75,0,0,0,1], model_path="" ):
        if model_path == "":
            model_path = os.path.expanduser(f'~/.gazebo/models/{model_name}/model.sdf')
        with open(model_path, 'r') as f:
            model_xml = f.read()
        pose = Pose()
        pose.position.x = target_pose[0]
        pose.position.y = target_pose[1]
        pose.position.z = target_pose[2]
        pose.orientation.x = target_pose[3]
        pose.orientation.y = target_pose[4]
        pose.orientation.z = target_pose[5]
        pose.orientation.w = target_pose[6]
        self.model_spawn_service(model_name, model_xml, '', pose, 'world')

    def delete_model(self, model_name):
        self.model_delete_service(model_name)

if __name__=="__main__":
    obj_service = objectService()
    # obj_service.set_state("fetch",[1,0,0,0,0,0,1])
    obj_service.spawn_model("cafe_table",[1,0,-0.03,0,0,0,1])
    rospy.sleep(1)
    obj_service.spawn_model('003_cracker_box',[1,0,0.75,0,0,0,1])
    rospy.sleep(2)

    obj_service.delete_model("cafe_table")
    obj_service.delete_model("003_cracker_box")