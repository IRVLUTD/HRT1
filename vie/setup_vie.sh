#!/bin/bash

#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

# Purpose: For vie eco-system setup

# Define the root directory
ROOT_DIR=$PWD

# Step 1: Install dependencies using conda and pip
echo "Installing PyTorch and CUDA dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia # should install based on CUDA_HOME

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Step 2: Install robokit (for gdino, samv2) and hydra-core
echo "Installing robokit and updating hydra-core..."
python setup.py install
pip install --upgrade --force-reinstall hydra-core # for samv2

# Step 3: Clone and install hamer at a specific commit
echo "Cloning and installing hamer repository..."
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
git checkout df533a2d04b9e2ece7cf9d6cbc6982e140210517 # checkout the specific commit
pip install -e .[all]

# Step 4: Download trained models
echo "Downloading demo data..."
bash fetch_demo_data.sh

# Step 5: Install ViTPose from the third-party directory
echo "Installing ViTPose from third-party/ViTPose..."
pip install -v -e third-party/ViTPose

# Step 6: Return to the root directory
echo "Returning to project root directory..."
cd $ROOT_DIR

echo "Installation complete!"

# Step 7: Reminder to download MANO models
echo "Please download the MANO models manually as described in the hamer README:"
echo "https://github.com/geopavlakos/hamer/tree/main?tab=readme-ov-file#installation"
