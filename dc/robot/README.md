### `catkin_ws` contains 
  - `ROS-TCP-endpoint` in `src/`
  - `python save_human_demo_data.py` in `scripts/`
  - `python sub_compress_pub.py task 0.1` in `scripts/`

### To start capture [Without HoloLens 2]
```shell
`rostopic pub /hololens/out/record_command std_msgs/Bool "data: true"`
```

### To stop capture [Without HoloLens 2]
```shell
rostopic pub /hololens/out/record_command std_msgs/Bool "data: false"
```

### Extra pointers
  - `depth` is multipled by `1000` and saved as `png`. Check [blog](https://jishnujayakumar.github.io/blog/2024/saving-depth-as-jpg-vs-png/) discussing why `png` and not `jpg`.
