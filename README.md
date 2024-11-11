
# MM-Demo
A codebase for XPENG 1-Shot Skill Learning, built on top of the [robokit](https://github.com/IRVLUTD/robokit) tool.

## Setup

```sh
# Clone the repository
git clone --recursive https://github.com/IRVLUTD/mm-demo && cd mm-demo

# Create a conda environment
conda create -n 1-shot python=3.10  # Python 3.10 required for samv2 and hamer dependencies
conda activate 1-shot

# Set your CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda

# Run the setup script
chmod +x ./setup_perception.sh
./setup_perception.sh

# todo: need to integrate hamer
cd hamer;
python demo.py \
--img_folder ../imgs/irvl-whiteboard-write-and-erase/ --out_folder irvl-whiteboard-write-and-erase-out \
--batch_size=48 --side_view --save_mesh --full_frame

# run
python perception_pipeline.py
```

### Tools

- To view `.obj` files: `python scripts/plot_obj.py <path/to/.obj>`
- Test gdino prompts: `python test_gdino_prompts.py --input_dir ./imgs/irvl-whiteboard-write-and-erase --text_prompt "black eraser"`
- Test gdino + samv2: `python test_gdino_samv2.py --input_dir="imgs/irvl-whiteboard-write-and-erase" --text_prompt="black eraser" --save_interval=1` 
  - Get bbox of the desired object from the first frame and then track them in the video frames using SAMv2  

### Acknowledgments

- [HPHB](https://github.com/IRVLUTD/HumanPoseHandBoxes)
- [GDINO + SamV2](https://github.com/IRVLUTD/robokit)
