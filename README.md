
# MM-Demo
A codebase for XPENG 1-Shot Skill Learning, built on top of the [robokit](https://github.com/IRVLUTD/robokit) framework.

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
```

### Tools

- To view `.obj` files: `python scripts/plot_obj.py <path/to/.obj>`

### Acknowledgments

- [HPHB](https://github.com/IRVLUTD/HumanPoseHandBoxes)
- [GDINO + SamV2](https://github.com/IRVLUTD/robokit)
