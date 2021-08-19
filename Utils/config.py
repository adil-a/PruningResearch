import random
import torch
import numpy
import os

# PRIVATE_PATH = '/ais/gobi3/u/adilasif/PruningResearch'  # TODO remove this before making repo public
# PRIVATE_PATH = os.getcwd()
PRIVATE_PATH = '/scratch/hdd001/home/gdzhang/adil_code/PruningResearch'
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
VGG_TARGET_SIZE = 5703649  # gotten from number of trainable parameters in VGG11 w/ expansion ratio of 1.0x times 0.2
RESNET_CIFAR_TARGET_SIZE = 274196  # gotten from number of params in ResNet20 model with 1.0 expansion ratio
SEED = 1
defaultcfg_vgg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

defaultcfg_resnet_imagenet = {
    18: [64, 128, 256, 512],
    34: [64, 128, 256, 512]
}

defaultcfg_resnet_cifar = {
    20: [16, 32, 64]
}


def setup_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
