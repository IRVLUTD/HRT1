catkin_ws contains 
ROS-TCP-endpoint in src
python save_human_demo_data related py scripts
python sub_compress_pub.py task 0.1

rostopic pub /hololens/out/record_command std_msgs/Bool "data: true" # start capture

rostopic pub /hololens/out/record_command std_msgs/Bool "data: false" # stop capture

depth is multipled by 1000 and saved as png
create a blog discussing why png and not jpg
