# HRT1
A codebase for Mobile Manipulation via One-Shot Human-to-Robot Trajectory Transfer, built on top of the [robokit](https://github.com/IRVLUTD/robokit) and [gto](https://github.com/IRVLUTD/GraspTrajOpt) tools.

- [`dc/`](dc/) contains the HoloLens app for data capture
- [`vie/`](vie/) contains human demo data capture and video information extraction (vie) modules
- [`dytto/`](dytto/) contains the logic for trajectory tracking optimization

## Setup

```sh
# Clone the repository
git clone --recursive https://github.com/IRVLUTD/HRT1 && cd HRT1

# Create a conda environment from scratch
conda create -n hrt1 python=3.10  # Python 3.10 required for samv2 and hamer dependencies
conda activate hrt1

# Set your CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda
```

- To get the latest changes from the submodules
```shell
git submodule sync
git submodule update --remote --merge --recursive
```