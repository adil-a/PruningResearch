# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/Models/lottery_vgg.py
# and https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
from Layers import layers
from Utils.config import defaultcfg
import torch
import math


class VGG(nn.Module):
    def __init__(self, dataset='cifar100', depth=11, init_weights=True, cfg=None, affine=True, batchnorm=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.features = self.make_layers(cfg, batchnorm)
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.dataset = dataset
        self.classifier = nn.Sequential(
            layers.Linear(cfg[-2], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            layers.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            layers.Linear(4096, num_classes)
        )
        # self.classifier = layers.Linear(cfg[-2], num_classes)
        # if init_weights:
        #     self._initialize_weights()

    def make_layers(self, cfg, batchnorm):
        layer_list = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'A':
                layer_list.append(nn.AvgPool2d(2))
            else:
                conv2d = layers.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batchnorm:
                    layer_list += [conv2d, layers.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layer_list += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_pruned_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, layers.Conv2d)):
                self._pruned_kaiming_normal_(m.weight, m.weight_mask)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _pruned_kaiming_normal_(self, tensor, mask, a=0, mode='fan_in', nonlinearity='relu'):
        fan = nn.init._calculate_correct_fan(tensor, mode)
        gain = nn.init.calculate_gain(nonlinearity, a)
        ratio_of_remaining_weights = torch.sum(torch.flatten(mask)).item() / torch.numel(mask)
        fan = int(fan * ratio_of_remaining_weights)
        std = gain / math.sqrt(fan)
        with torch.no_grad():
            return tensor.normal_(0, std)
