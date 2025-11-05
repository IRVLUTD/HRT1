# 🚀 Trajectory Tracking Optimization 🌟

---

## 📦 Install the Module


### 1. Create and Activate Conda Environment 🐍
Set up a Python 3.9 environment using Conda for dependency management.
```
conda create -n trajopt python=3.9
conda activate trajopt
```

### 2. Install Dependencies 📚
Install ROS and Python dependencies required for the project.
```
source /opt/ros/noetic/setup.bash
chmod +x install_ros_deps.sh
sh install_ros_deps.sh
pip install -r requirements.txt
```

### 3. Build the ROS Workspace 🏗️
Navigate to the workspace and build it with the following commands.
```
cd mm_ws
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y
catkin clean --yes
catkin_make -DCMAKE_BUILD_TYPE=Release
```

> **⚠️ Note**: If you encounter errors related to `LIBFFI` version, refer to this [issue](https://github.com/IRVLUTD/trajectory-tracking-optimization/issues/1) for guidance.

---

## 🎮 Running in Simulation

### 1. Launch Gazebo Environment 🌍
Start the Gazebo simulation with a GUI for visualization.
```
roslaunch aws_robomaker_small_house_world small_house.launch gui:=True
```

### 2. Launch Robot 🤖
Spawn the robot in the Gazebo environment.
```
roslaunch fetch_gazebo spawn_robot.launch
```
Make sure to use correct ```fetch.urdf``` according to desired gripper config

### 3. Launch MoveIt 🦾
Initialize MoveIt for motion planning. If using gripper config 1:
```
roslaunch fetch_moveit_config moveit.launch robot:=fetch_original
```
if using gripper config 2:
```
roslaunch fetch_moveit_config move_group.launch
```

### 4. Launch RViz 📊
Visualize the robot and environment in RViz.
```
cd mm_ws/config
rosrun rviz rviz -d tto.rviz
```
### 5. Update Task and robot params
```
cd mm_ws/scripts/config/
```
Update ```paths.py``` to reflect the ```TASK_ID```, Gripper configs, etc..,. 
NOTE: ```CURRENT_GRIPPER_CONFIG``` should always be 0. 

### 6. Run the GroundingSAM ROS server
This script is part of the vie module and is present in ```mm-demo/vie/```.
```
python gsam_server.py
```

### 7. Run Optimization ⚙️

Run trajectory optimization including the robot's base.
```
cd mm_ws/scripts/traj_opt
python run.py --stow_dir "right" --obj_prompt <object-name> 
```
All the key params are taken from ```paths.py```. Make sure to have it updated relevant to the task being conducted.


**Note**: For running in realworld, just set the param ```IS_SIM=False```. 

