# 🚀 Trajectory Tracking Optimization 🌟

This repository provides a complete pipeline for **trajectory tracking and optimization** for **mobile manipulation tasks**, supporting both **simulation** and **real-world execution** using the **Fetch robot** on **ROS Noetic**.

---

## 🗂️ **Index**

1. [📦 Install the Module](#-install-the-module)
   - [🐍 Create and Activate Conda Environment](#1-create-and-activate-conda-environment-)
   - [📚 Install Dependencies](#2-install-dependencies-)
   - [🏗️ Build the ROS Workspace](#3-build-the-ros-workspace-)
2. [🎮 Running in Simulation](#-running-in-simulation)
   - [🌍 Launch Gazebo Environment](#1-launch-gazebo-environment-)
   - [🤖 Launch Robot](#2-launch-robot-)
   - [🦾 Launch MoveIt](#3-launch-moveit-)
   - [📊 Launch RViz](#4-launch-rviz-)
   - [⚙️ Update Task and Robot Params](#5-update-task-and-robot-params)
   - [🧠 Run GroundingSAM ROS Server](#6-run-the-groundingsam-ros-server)
   - [🚀 Run Optimization and Task Execution (Simulation)](#7-run-optimization-and-task-execution-in-sim)
3. [🏠 Running in Real World](#-running-in-realworld)
   - [🧩 Runtime Object Pose Transformation (BundleSDF)](#7-run-time-object-pose-transformation-realtime)
   - [🚀 Run Optimization and Task Execution (Real World)](#8-run-optimization-and-the-task-execution-realworld)


---
https://github.com/user-attachments/assets/3359c6ae-18cd-4906-8600-5a16a5120054

## 📦 Install the Module

### 1. Create and Activate Conda Environment 🐍
Set up a Python 3.9 environment using Conda for dependency management.

```bash
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

> **⚠️ Note**: If you encounter errors related to `LIBFFI` version, refer to this [issue](https://github.com/eclipse-sumo/sumo/issues/14773#issuecomment-2466751518) for guidance.

---

## 🎮 Running in Simulation
The runtime mainly involves spawning the environment, robot, GSAM server, Object Pose estimation, Mobile base and Joint trajectory optimization to execute the task.

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
Make sure to use correct ```fetch.urdf``` according to desired gripper config. The default urdf file corresponds to config 1. 

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
This GroundingSAM (GSAM) service is part of the vie module and is present in ```HRT1/vie```. Please ensure to activate the [robokit environment](../vie/README.md##Requirements) to run this service.
```
python gsam_server.py
```
Optimization requires the target object mask, so that it will not consider the object pointcloud as obstacle while tracking the trajectory. so we run this GSAM ROS service to query object mask with the prompt being passed as argument in the main script next. Set ```IS_MASK=True``` in ```paths.py```. If you want to test the optimization without it, no need to run this server and set ```IS_MASK=False```.


🧱 Note: 
Steps 1–6 only need to be launched once.
Step 7 (Optimization) can be rerun for each new task or object.

### 7. Run Optimization and task execution in Sim

Run trajectory optimization including the robot's base. This script automatically spawn the scene corresponding to "move the cracker box" task. If you want to run with base optimization, set ```IS_BASE=True```.

🧱 NOTE: It is an approximate scene setup compared to the actual realworld demonstration, and is intended for users to test and validate the optimization module before deploying on real hardware.

```
cd mm_ws/scripts/traj_opt
python run.py --stow_dir "right" --obj_prompt <object-name> 
```
All the key params are taken from ```paths.py```. Make sure to have it updated relevant to the task being conducted.


## 🏠 Running in realworld
**Note**: For running in realworld, just set the param ```IS_SIM=False``` and ```IS_DELTA=True```. 
run steps 1-6 from the [ Running in Simulation](#-running-in-simulation) section, before proceedinf further.

### 7. Run time Object Pose transformation realtime
To estimate the object pose during run time, relative to demonstration first frame, we use BundleSDF. This aligns the real-world scene with the recorded demonstration frames to provide accurate object-relative transformations.

```
cd vie/docker/
./enter_docker.sh && ./start_docker.sh
cd ..
./run_bundlesdf.sh <path to the specific task demonstration folder> 10 5. where 10 indicates the number of demo frames and 5 indicates the number of rollout frames to be used.
```

### 8. Run Optimization and the task execution realworld

Run trajectory optimization including the robot's base.
```
cd mm_ws/scripts/traj_opt
python run.py --stow_dir "right" --obj_prompt <object-name> 
```
All the key params are taken from ```paths.py```. Make sure to have it updated relevant to the task being conducted.








