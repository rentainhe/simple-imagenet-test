import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'results'
# GPU Settings, overwritten by command line argument, If you want to use multi-gpu testing, you can set e.g. '0, 1, 2' instead
_C.GPU = '0'
_C.N_GPU = None

# -----------------------------------------------------------------------------
# Testing Data settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.TEST.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.TEST.DATA_PATH = ''
# Path to label, could be overwritten by command line argument
_C.TEST.LABEL_PATH = './val_label.txt'
# Dataset name
_C.TEST.DATASET = 'imagenet'
# Input image size
_C.TEST.IMG_SIZE = 384
# Interpolation to resize image (random, bilinear, bicubic)
_C.TEST.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.TEST.PIN_MEMORY = True
# Number of data loading threads
_C.TEST.NUM_WORKERS = 8
# Whether to use center crop when testing
_C.TEST.CROP = True
# Imagenet Default Mean
_C.TEST.MEAN = (0.485, 0.456, 0.406)
# Imagenet Default Std
_C.TEST.STD = (0.229, 0.224, 0.225)

def update_config(config, args):
    config.deforst()
    if args.batch_size:
        config.TEST.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.TEST.DATA_PATH = args.data_path
    if args.label_path:
        config.TEST.LABEL_PATH = args.label_path
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.TEST.DATASET, config.TAG)
    # setup devices
    if config.GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
        config.N_GPU = len(config.GPU.split(','))
        config.DEVICES = [_ for _ in range(config.N_GPU)]
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config