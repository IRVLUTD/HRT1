### `catkin_ws` contains 
  - `ROS-TCP-endpoint` in `src/`
  - `python save_human_demo_data.py --slop_seconds 0.3` in `scripts/`
  - Capture data will be saved in `scripts/data_captured`
  - Any number of task demos can be recorded.
    - No need to run the scripts many times.
    - Run the script untill all the required tasks are recorded.

### To test start capture [Without HoloLens 2]
```shell
rostopic pub /hololens/out/record_command std_msgs/Bool true
```

### To test stop capture [Without HoloLens 2]
```shell
rostopic pub /hololens/out/record_command std_msgs/Bool false
```

### Extra pointers
  - `depth` is multipled by `1000` and saved as `png`. Check [blog](https://jishnujayakumar.github.io/blog/2024/saving-depth-as-jpg-vs-png/) discussing why `png` and not `jpg`.
