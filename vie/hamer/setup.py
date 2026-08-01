from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='hamer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'torch',
        'torchvision',
        'yacs',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        'chumpy @ git+https://github.com/mattloper/chumpy',
        # mmpose 0.24.0 only accepts mmcv in [1.3.8, 1.5.0]; the original 1.3.9
        # pin builds with very old setuptools and breaks on newer Python toolchains.
        # 1.5.0 is the highest compatible with mmpose 0.24 and installs cleanly.
        'mmcv>=1.3.8,<=1.5.0',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
            'webdataset',
        ],
    },
)
