# RoboKit
A toolkit for robotic tasks

## Features
- Zero-shot text-to-bbox approach for object detection using GroundingDINO.
- Zero-shot bbox-to-mask approach for object detection using SegmentAnything (MobileSAM).
- Zero-shot classification using OpenAI CLIP.

## Getting Started

### Prerequisites
TODO
- Python 3.7 or higher (tested 3.9)
- torch (tested 2.0)
- torchvision

### Installation
```sh
git clone https://github.com/IRVLUTD/robokit.git && cd robokit 
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python setup.py install
```

Note: Check GroundingDINO [installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for the following error
```sh
NameError: name '_C' is not defined
```

## Usage
- SAM: [`test_sam.py`](test_sam.py)
- GroundingDINO + SAM: [`test_gdino_sam.py`](test_gdino_sam.py)
- GroundingDINO + SAM + CLIP: [`test_gdino_sam_clip.py`](test_gdino_sam_clip.py)
- Test Datasets: [`test_dataset.py`](test_dataset.py)
  - `python test_dataset.py --gpu 0 --dataset <ocid_object_test/osd_object_test>`

## Roadmap

Future goals for this project include: 
TODO

## Acknowledgments

This project is based on the following repositories:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [CLIP](https://github.com/openai/CLIP)

## License
This project is licensed under the MIT License