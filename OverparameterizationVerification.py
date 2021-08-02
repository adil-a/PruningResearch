import argparse
import os

from Utils.config import PRIVATE_PATH, BATCH_SIZE, EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, SEED, defaultcfg
from Utils.network_utils import get_network, multiplier, dataloader
from train import train
from Optimizers.lars import LARS

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch


def del_file(path):
    if os.path.isfile(path):
        os.remove(path)


def get_lr(ratio: float):
    new_lr = LR / ratio
    if new_lr > 0.1:
        return 0.1
    else:
        return new_lr


def main(args):
    print(f'Number of GPUs being used: {torch.cuda.device_count()}')
    torch.manual_seed(SEED)

    current_ratio = args.expansion_ratio
    current_lr = args.lr
    current_cfg = defaultcfg[11]
    multiplier(current_cfg, current_ratio)

    trainloader = dataloader('cifar100', BATCH_SIZE, True)
    testloader = dataloader('cifar100', BATCH_SIZE, False)
    # for lr in lr_array:
    print(f'Current VGG11 config being used: {current_cfg} (ratio {current_ratio}x) (Batchsize: {BATCH_SIZE}, '
          f'LR: {current_lr})')
    saved_file_name = f'vgg11_{current_ratio}x_{current_lr}'
    PATH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/{saved_file_name}_best.pt'
    PATH_FINAL_EPOCH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/' \
                                      f'{saved_file_name}_final_epoch.pt'
    if not os.path.isdir(PRIVATE_PATH + '/Models/SavedModels/expansion_ratio_inference/'):
        os.mkdir(PRIVATE_PATH + '/Models/SavedModels/expansion_ratio_inference/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # writer = SummaryWriter(f'runs/CIFAR100/VGG/{saved_file_name}')
    wandb.login()
    config = dict(learning_rate=args.lr,
                  dataset=args.dataset,
                  model=args.model_name,
                  train_epochs=args.post_epochs)
    wandb.init(project='Pruning Research',
               config=config,
               entity='sparsetraining')
    wandb.run.name = saved_file_name
    wandb.run.save()

    net = get_network('vgg11', 'cifar100', current_cfg, imp=False)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train(net, trainloader, testloader, optimizer, scheduler, criterion, device, None, PATH,
          PATH_FINAL_EPOCH, args.post_epochs, args.checkpoint_dir)
