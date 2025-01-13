# MM-Demo
A codebase for XPENG 1-Shot Skill Learning, built on top of the [robokit](https://github.com/IRVLUTD/robokit) and [gto](https://github.com/IRVLUTD/GraspTrajOpt) tools.

- [`dc/`](dc/) contains the HoloLens app for data capture
- [`vie/`](vie/) contains human demo data capture and video information extraction (vie) modules


## Setup

```sh
# Clone the repository
git clone --recursive https://github.com/IRVLUTD/mm-demo && cd mm-demo

# Create a conda environment from scratch
conda create -n 1-shot python=3.10  # Python 3.10 required for samv2 and hamer dependencies
conda activate 1-shot

# Set your CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda
