# mm-demo
Codebase for XPENG 1-Shot Skill Learning

## Setup
```sh
# clone
git clone --recursive https://github.com/IRVLUTD/mm-demo && cd mm-demo

# create conda env
conda create -n 1-shot python=3.9
conda activate 1-shot

# make sure your CUDA_HOME env var is set
export CUDA_HOME=/usr/local/cuda

root_dir=$PWD

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia # should install based on CUDA_HOME
pip install -r requirements.txt

# install robokit for gdino, samv2
python setup.py install
pip install --upgrade --force-reinstall hydra-core

cd $root_dir
```


### Acknowledgments
- [HPHB](https://github.com/IRVLUTD/HumanPoseHandBoxes)
- [GDINO + SamV2](https://github.com/IRVLUTD/robokit)