cmake_minimum_required(VERSION 3.0.2)
project(my_ros_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  tf2
  tf2_ros
  message_filters
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/run_ros_node.py
  scripts/data_collection_node.py
  scripts/config_manager.py
  scripts/frame_processor.py
  scripts/image_publisher.py
  scripts/bundlesdf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
