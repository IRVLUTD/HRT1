# RoboKit
A toolkit for robotic tasks

## Features
- Zero-shot classification using OpenAI CLIP.
- Zero-shot text-to-bbox approach for object detection using GroundingDINO.
- Zero-shot bbox-to-mask approach for object detection using SegmentAnything (MobileSAM).
- Zero-shot image-to-depth approach for depth estimation using Depth Anything.
- Zero-shot feature upsampling using FeatUp.
- Zero-shot DoorHandle detection using [iTeach](https://irvlutd.github.io/iTeach/)-[DHYOLO](https://huggingface.co/spaces/IRVLUTD/DH-YOLO) model
- Zero-shot bbox-to-mask video propogation approach for object tracking using SegmentAnythingV2 (SAMv2).
  - Note that SAMv2 only support mp4 or jpg files as of 11/06/2024
  - Currently only supports bbox prompt with all video frames stored as jpg files in a directory
  - If you have an mp4 file then extract individual frames as jpg and store in a directory


## Getting Started

### Prerequisites
TODO
- Python 3.7 or higher (tested 3.9.18)
- torch (tested 2.0)
- torchvision
- pytorch-cuda=11.8 (tested)
- [SAMv2 requires py>=3.10.0](https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/setup.py#L171) (here the installation has been tweaked to remove this constraint)

### Installation
```sh
# clone
git clone https://github.com/IRVLUTD/robokit.git && cd robokit 

# make sure your CUDA_HOME env var is set
export CUDA_HOME=/usr/local/cuda

# install dependencies
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install
python setup.py install
```

### Known Installation Issues 
- Check GroundingDINO [installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for the following error
```sh
NameError: name '_C' is not defined
```
- For SAMv2, `ModuleNotFoundError: No module named 'omegaconf.vendor'`
```sh
pip install --upgrade --force-reinstall hydra-core
```

## Usage
- Note: All test scripts are located in the [`test`](test) directory. Place the respective test scripts in the root directory to run.
- SAM: [`test_sam.py`](test/test_sam.py)
- GroundingDINO + SAM: [`test_gdino_sam.py`](test/test_gdino_sam.py)
- GroundingDINO + SAM + CLIP: [`test_gdino_sam_clip.py`](test/test_gdino_sam_clip.py)
- Depth Anything: [`test_depth_anything.py`](test/test_depth_anything.py)
- FeatUp: [`test_featup.py`](test/test_featup.py)
- iTeach-DHYOLO: [`test_dhyolo.py`](test/test_dhyolo.py)
- SAMv2: [`test_samv2.py`](test/test_samv2.py)
- Test Datasets: [`test_dataset.py`](test/test_dataset.py)
  - `python test_dataset.py --gpu 0 --dataset <ocid_object_test/osd_object_test>`

## Roadmap

Future goals for this project include: 
- Add a config to set the pretrained checkpoints dynamically
- More: TODO

## Acknowledgments

This project is based on the following repositories (license check mandatory):
- [CLIP](https://github.com/openai/CLIP)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [DepthAnything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation)
- [FeatUp](https://github.com/mhamilton723/FeatUp)
- [iTeach](https://irvlutd.github.io/iTeach/)-[DHYOLO](https://huggingface.co/spaces/IRVLUTD/DH-YOLO)
- [SAMv2](https://github.com/facebookresearch/sam2)


## License
This project is licensed under the MIT License. However, before using this tool please check the respective works for specific licenses.
