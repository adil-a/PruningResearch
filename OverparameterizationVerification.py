import argparse
import os

from Utils.config import PRIVATE_PATH, BATCH_SIZE, EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, SEED
from Utils.network_utils import get_network, multiplier, dataloader
from train import train
from Optimizers.lars import LARS

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch

FIND_BASELINE = False  # used in validation to find a baseline that fully fits our training data
defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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
    # lr_array = get_lr_array(0.1 / current_ratio, 0.1)
    # print(f'List of LRs: {lr_array}')
    current_cfg = defaultcfg[11]
    multiplier(current_cfg, current_ratio)

    # old_dr_best = ''
    # old_dr_final = ''
    # top_acc = 0
    # best_lr = 0
    # best_lr_acc = 0

    trainloader = dataloader('cifar100', BATCH_SIZE, True)
    testloader = dataloader('cifar100', BATCH_SIZE, False)
    # for lr in lr_array:
    print(f'Current VGG11 config being used: {current_cfg} (ratio {current_ratio}x) (Batchsize: {BATCH_SIZE}, '
          f'LR: {current_lr})')
    saved_file_name = f'vgg11_{current_ratio}x'
    PATH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/{saved_file_name}_best.pt'
    PATH_FINAL_EPOCH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/' \
                                      f'{saved_file_name}_final_epoch.pt'
    if not os.path.isdir(PRIVATE_PATH + '/Models/SavedModels/expansion_ratio_inference/'):
        os.mkdir(PRIVATE_PATH + '/Models/SavedModels/expansion_ratio_inference/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f'runs/CIFAR100/VGG/{saved_file_name}')

    net = get_network('vgg11', 'cifar100', current_cfg, imp=False)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    # optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    #                       nesterov=True)
    optimizer = LARS(net.parameters(), lr=args.lr, max_epoch=args.post_epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if FIND_BASELINE:
        train(net, trainloader, trainloader, optimizer, scheduler, criterion, device, writer, PATH,
                    PATH_FINAL_EPOCH, args.post_epochs, args.checkpoint_dir)
    else:
        train(net, trainloader, testloader, optimizer, scheduler, criterion, device, writer, PATH,
                    PATH_FINAL_EPOCH, args.post_epochs, args.checkpoint_dir)
    # if acc > top_acc:
    #     top_acc = acc
    #     best_lr = lr
    #     best_lr_acc = val(net, testloader, device, None)[0]  # this is evaluated on the final model and not the best
    #     # performing model
    #     del_file(old_dr_best)
    #     del_file(old_dr_final)
    #     old_dr_best = PATH
    #     old_dr_final = PATH_FINAL_EPOCH
    # else:
    #     del_file(PATH)
    #     del_file(PATH_FINAL_EPOCH)
    # print(f'Best LR is {best_lr} with test accuracy {best_lr_acc}')
