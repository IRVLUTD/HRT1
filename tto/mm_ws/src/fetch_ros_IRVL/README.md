# fetch_ros_IRVL

ROS Components for Robots from Fetch Robotics used at the Intelligent Robotics and Vision Lab at the University of Texas at Dallas. Our Fetch Robot uses an ATI-Gamma F/T sensor at the wrist; in this forked repository you will find the updated components for our robot, such as the new .urdf, moveit configuration, and a ros package to interface with the sensor. 

### Table of Contents

- [fetch\_ros\_IRVL](#fetch_ros_irvl)
    - [Table of Contents](#table-of-contents)
  - [New .Urdf](#new-urdf)
    - [Moveit Configuration](#moveit-configuration)
    - [Robot Calibration](#robot-calibration)
      - [Notes](#notes)
  - [fetch\_netft package](#fetch_netft-package)
    - [Notes](#notes-1)
  - [New Deformable Fingers](#new-deformable-fingers)
- [fetch\_ros Original Readme.md](#fetch_ros-original-readmemd)
  - [Documentation](#documentation)
  - [ROS Buildfarm Development Branches](#ros-buildfarm-development-branches)
  - [ROS Buildfarm Release](#ros-buildfarm-release)

## New .Urdf 

As mentioned before, our Fetch robot is equipped with an ATI-Gamma force and torque sensor at the wrist. Consequently, the original .urdf can no longer be used for the proper control of the robot. We created a new .urdf [fetch.urdf](/fetch_description/robots/fetch.urdf) to replace the original at the fetch_description base ros package located within the '/opt/ros/' directory. 

<p align="center">
<img src="./media/robot.jpeg"  height="500" alt="IRVL Fetch Robot">
</p>

Replacing the original .urdf file will make the robot boot with the new robot representation which will respect the new joint limits when using teleoperation. In the case that the robot has been calibrated the change will only take effect if the calibration is reseted to it's factory model. You can do this by running:

```Shell
    calibrate_robot --reset
```

Check if the change took effect by running 
```Shell
    calibrate_robot --date
```
The output should be 'uncalibrated'.

### Moveit Configuration
Since we have updated the .urdf file we also made modifications to the [fetch_moveit_config](/fetch_moveit_config/) package. First, we added the new 'ati_link' to the [fetch.srdf](/fetch_moveit_config/config/fetch.srdf) for the simplification of the MoveIt path planning. Second, we reduced the joint velocity limits in [joint_limits.yaml](/fetch_moveit_config/config/joint_limits.yaml) to more reasonable speeds.

After the changes have been made, ensure they are loaded by restarting the roscore service. NOTE: The fetch_drivers will reset; place the fetch robot in a resting position! (arm tucked, torso down) before running the command.

```Shell
    sudo service roscore restart
```

### Robot Calibration
To calibrate our updated robot we had to make some adjustments to the files used in the calibration process referred to at the [Fetch and Freight manual](https://docs.fetchrobotics.com/calibration.html). The calibration robot process uses the RGB-D camera to detect the positions of 4 LEDs in the Gripper while performing 99 different poses. To calibrate the new robot model we had to: 

1) Modify the calibration poses in [calibration_poses.bag](/fetch_calibration/config/calibration_poses.bag) to ensure that the LEDs are visible by the RGB-D camera. This was done using the script [modify_poses.py](/fetch_calibration/config/modify_poses.py) by adding an offset to the *wrist_roll_joint*. 
2) The 4 LED positions were updated in [capture.yaml](/fetch_calibration/config/capture.yaml). The original positions are with respect to the original *wrist_roll_joint*; we used a simple transformation to devise the new positions. [point_angle_rotation.py](/fetch_calibration/config/point_angle_rotation.py)
3) After modifying the files you can follow the instructions in the [Fetch and Freight manual](https://docs.fetchrobotics.com/calibration.html) to calibrate. Note that after running the command:
```Shell
    calibrate_robot --arm --install --velocity-factor=0.5
```
If you have sudo privilege the robot drivers will restart immediately after performing the calibration. i.e. the arm will become unactuated and fall. We also recommend using the *velocity-factor* argument to reduce the speed of the robot.

#### Notes
- The tuck_arm routine [tuck_arm.py](/fetch_teleop/scripts/tuck_arm.py) was modified to account for our new robot model and to ensure no collisions with the installed equipment. For our robot the script was located within the '/opt/ros/melodic/lib/fetch_teleop/' directory.
- We included our calibrated model in [calibrated_files](/fetch_description/robots/calibrated/).

## fetch_netft package
To interface with the ATI sensor NetFT device, we developed a ROS package that let's you activate the NetFT UDP/RDT data stream and publish the values to a rostopic. We assigned the NetFT the IP address: 10.42.42.41 on the Fetch internal network. The [sensor.py](/fetch_netft/scripts/sensor.py) script connects to the NetFT and publishes the wrench values as `WrenchStamped` in the 4 different rostopics. You can initialize the sensor data stream by running the launch file:

```Shell
roslaunch fetch_netft netft.launch
```

The published Wrench values are with respect to our fetch.urdf `ati_link`.

<p align="center">
<video src="media/sensor_video.mp4" controls></video>
</p>

The four rostopics are:
- `gripper/ft_sensor/raw` for raw values sensed by the force torque sensor (software bias is set when initialized).
- `gripper/ft_sensor/absolute` uses the gripper information to publish the absolute force and torque sensed by the force sensor. 
- `gripper/ft_sensor/external` the calculated external forces on the sensor after accounting for the gripper weight and CoG at the current gripper pose.
- `gripper/ft_sensor/external_imu` calculated external forces using the gripper IMU directly.

### Notes

- You can find the UDP interface documentation in the [NetFT Manual](https://www.ati-ia.com/app_content/documents/9610-05-1022.pdf).

## New Deformable Fingers
We have also changed the original Fetch fingers to a adapted version of the [UMI deformable fingers](https://umi-gripper.github.io/). Both .urdfs modeling the original fingers and the new deformable fingers are included in the [fetch_description](/fetch_description/robots/) package. 

<p align="center">
<img src="./media/fingers.jpeg" height="400" alt="Deformable Fingers Image">
</p>

# fetch_ros Original Readme.md

Open ROS Components for Robots from Fetch Robotics

## Documentation

Please refer to our documentation page: http://docs.fetchrobotics.com/

## ROS Buildfarm Development Branches

Fetch Package | Indigo Devel | Kinetic Devel | Melodic Devel | ROS 1 (Noetic) | Dashing Devel
------------- | ------------ | ------------- | ------------- | -------------- | -------------
fetch_ros |  EOL | :negative_squared_cross_mark: not supported | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mdev__fetch_ros__ubuntu_bionic_amd64)](http://build.ros.org/job/Mdev__fetch_ros__ubuntu_bionic_amd64/) | :hammer_and_wrench: forthcoming :hammer_and_wrench: | :construction: in planning :construction:

Kinetic support, has been skipped. Ubuntu 18.04 and 20.04 support both ROS1 and ROS2, so it was decided that effort will go towards ROS2 support in the future instead of supporting Kinetic. Noetic support is in the works, also.

## ROS Buildfarm Release

Fetch Package | Indigo | Kinetic * | Melodic Source | Melodic Debian
------------- | ------ | --------- | -------------- | --------------
fetch_calibration | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_calibration__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_calibration__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_calibration__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_calibration__ubuntu_bionic_amd64__binary/)| 
fetch_depth_layer | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_depth_layer__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_depth_layer__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_depth_layer__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_depth_layer__ubuntu_bionic_amd64__binary/) |
fetch_description | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_description__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_description__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_description__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_description__ubuntu_bionic_amd64__binary/) |
fetch_ikfast_plugin | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_ikfast_plugin__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_ikfast_plugin__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_ikfast_plugin__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_ikfast_plugin__ubuntu_bionic_amd64__binary/) |
fetch_maps | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_maps__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_maps__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_maps__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_maps__ubuntu_bionic_amd64__binary/) | :negative_squared_cross_mark: | 
fetch_moveit_config | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_moveit_config__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_moveit_config__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_moveit_config__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_moveit_config__ubuntu_bionic_amd64__binary/) | 
fetch_navigation | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_navigation__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_navigation__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_navigation__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_navigation__ubuntu_bionic_amd64__binary/) |
fetch_teleop | EOL | :negative_squared_cross_mark: | [![Build Status](http://build.ros.org/buildStatus/icon?job=Msrc_uB__fetch_teleop__ubuntu_bionic__source)](http://build.ros.org/job/Msrc_uB__fetch_teleop__ubuntu_bionic__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Mbin_uB64__fetch_teleop__ubuntu_bionic_amd64__binary)](http://build.ros.org/job/Mbin_uB64__fetch_teleop__ubuntu_bionic_amd64__binary/) |
