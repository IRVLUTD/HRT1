### 📂 `catkin_ws` contains:
- 📡 `ROS-TCP-endpoint` in `src/`
- 🐍 `python save_human_demo_data.py --slop_seconds 0.3` in `scripts/`
- 💾 Captured data saved in `scripts/data_captured/`
- 🔁 Supports recording **multiple task demos**:
  - ✅ No need to restart the script each time
  - ▶️ Just **keep it running** until all required tasks are captured

---

### 🧪 To start capture [Without HoloLens 2]
```shell
rostopic pub /hololens/out/record_command std_msgs/Bool true
```

---

### 🧪 To stop capture [Without HoloLens 2]
```shell
rostopic pub /hololens/out/record_command std_msgs/Bool false
```

---

### 💡 Extra pointers:
- 🌊 `depth` is multiplied by `1000` and saved as `.png`
- 📚 See [this blog](https://jishnujayakumar.github.io/blog/2024/saving-depth-as-jpg-vs-png/) for **why PNG > JPG** for depth data
