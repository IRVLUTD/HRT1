#!/bin/bash
# build_ros_package.sh
# Copies vie/BundleSDF to ROS workspace and builds it

# Set variables
ROS_WS=/catkin_ws
SRC_DIR=$PWD
DST_DIR=$ROS_WS/src/my_ros_package

# Ensure ROS Noetic is sourced
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
else
    echo "Error: ROS Noetic setup.bash not found at /opt/ros/noetic/setup.bash"
    exit 1
fi

# Create workspace if it doesn't exist
mkdir -p $ROS_WS/src

# Copy BundleSDF to workspace
if [ -d "$SRC_DIR" ]; then
    echo "Copying $SRC_DIR to $DST_DIR..."
    cp -r $SRC_DIR $DST_DIR
else
    echo "Error: Source directory $SRC_DIR does not exist"
    exit 1
fi

# Build the workspace
cd $ROS_WS
echo "Building ROS workspace with catkin_make..."
if ! /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"; then
    echo "Error: catkin_make failed"
    exit 1
fi

echo "Build completed successfully"