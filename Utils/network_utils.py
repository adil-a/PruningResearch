# loader code taken from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
import numpy

from Models import VGGModels
from typing import List, Union

import torch
import numpy as np
import math

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

ERROR_MESSAGE = 'Invalid network name'


def get_train_valid_loader(batch_size,
                           val_set,
                           valid_size=0.1,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-100 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomRotation(15)]
    )

    train_dataset = datasets.CIFAR100(
        root='./data', train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root='./data', train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    if val_set:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        valid_loader = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=pin_memory, shuffle=True
        )

    return train_loader, valid_loader


def get_test_loader(batch_size,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    dataset = datasets.CIFAR100(
        root='./data', train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_network(name: str, dataset: str, config: List[Union[int, str]]):
    ds = dataset.lower()
    net_name = name.lower()
    if net_name.startswith('vgg'):
        depth = int(net_name.replace('vgg', ''))
        if depth != 11 and depth != 13 and depth != 16 and depth != 19:
            print(ERROR_MESSAGE)
        else:
            return VGGModels.VGG(depth=depth, dataset=ds, cfg=config)
    else:
        print(ERROR_MESSAGE)


def multiplier(cfg: List[Union[int, str]], ratio: float):
    for i in range(len(cfg)):
        if isinstance(cfg[i], int):
            cfg[i] = int(ratio * cfg[i])


def get_lr_array(start, stop):
    temp_arr = np.logspace(start, stop, num=5, base=10)
    lst = []
    sig_figs = 3
    if start >= stop:
        lst.append(stop)
        return lst
    for value in temp_arr:
        lr = np.log10(value)
        lst.append(round(lr, sig_figs - int(math.floor(math.log10(abs(lr)))) - 1))
    return lst
