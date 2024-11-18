# Run the setup script
chmod +x ./setup_perception.sh
./setup_perception.sh

# run (todo hbhp no longer needed as hamer has hand bbox detection enabled)
python perception_pipeline.py # hbhp+gdino+samv2
```

## Tools
#### Note: To run, move all relevant scripts inside `scripts/` to project root folder
- To view `.obj` files: 
  ```shell
  python scripts/vis_obj.py <path/to/.obj>
  ```

- Test gdino prompts:
  ```shell
  python test_gdino_prompts.py --input_dir ./imgs/irvl-whiteboard-write-and-erase/rgb --text_prompt "black eraser"
  # Output gets saved in `./imgs/gdino/irvl-whiteboard-write-and-erase/black_eraser`
  ```

- Test gdino + samv2: (Get bbox of the desired object from the first frame and then track them in the video frames using SAMv2)
  ```shell
  python test_gdino_samv2.py --input_dir ./imgs/irvl-whiteboard-write-and-erase/rgb --text_prompt "black eraser" --save_interval=1
  # Output gets saved in 
  # `./imgs/irvl-whiteboard-write-and-erase/samv2/black_eraser/masks` mask overlayed + init obj bbox
  # `./imgs/irvl-whiteboard-write-and-erase/samv2/black_eraser/traj_overlayed` trajectory + mask overlayed + init obj bbox
  ```

- Test hamer:
  ```shell
  cd hamer
  python demo.py --img_folder ../imgs/irvl-whiteboard-write-and-erase/rgb/ \
  --out_folder irvl-whiteboard-write-and-erase-test \
  --batch_size=48 --side_view --save_mesh --full_frame
  ``` 

- Get right/left hand bboxes and meshes: [assumption: only one person exists in the scene]
  ```shell
  cd  hamer
  python extract_hand_bboxes_and_meshes.py --input_dir "../imgs/irvl-whiteboard-write-and-erase/rgb/"
  # Output gets saved in `./imgs/irvl-whiteboard-write-and-erase/hamer/` (.obj, .png)
  ```

### Acknowledgments

- [HPHB](https://github.com/IRVLUTD/HumanPoseHandBoxes)
- [GDINO + SamV2](https://github.com/IRVLUTD/robokit)
