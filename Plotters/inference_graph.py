import os
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch

from Utils.config import PRIVATE_PATH, BATCH_SIZE, SEED, defaultcfg_resnet_cifar, defaultcfg_vgg, defaultcfg_resnet_imagenet
from Utils.network_utils import get_network, multiplier, dataloader, eval


def load_for_non_parallel(dictionary):
    new_state_dict = OrderedDict()
    for k, v in dictionary.items():
        temp = k[:7]
        name = k[7:]
        # print(name)
        if temp == 'module.':
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_file_names(ratios):
    lst = []
    all_files = os.listdir(PRIVATE_PATH + '/Models/SavedModels/VGG/expansion_ratio_inference/')
    for ratio in ratios:
        for file in all_files:
            if str(ratio) in file and 'best' in file:
                lst.append(file)
    return lst


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = dataloader(args.dataset.lower(), BATCH_SIZE, False)
    if args.model_name.lower() == 'vgg11':
        RATIOS = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    elif args.model_name.lower() == 'resnet20':
        RATIOS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    accuracies = []
    for ratio in RATIOS:
        print(f'{ratio}x')
        if 'vgg' in args.model_name.lower():
            current_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
        else:
            if args.dataset.lower() == 'imagenet':
                current_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
            else:
                current_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
        multiplier(current_cfg, ratio)
        file_to_open = f'{args.model_name.lower()}_{ratio}x_best.pt'
        if 'vgg' in args.model_name.lower():
            PATH = PRIVATE_PATH + f'/Models/SavedModels/VGG/expansion_ratio_inference/{file_to_open}'
        elif 'resnet' in args.model_name.lower():
            PATH = PRIVATE_PATH + f'/Models/SavedModels/ResNet/expansion_ratio_inference/{file_to_open}'
        net = get_network(args.model_name, args.dataset, current_cfg)
        state_dict = load_for_non_parallel(torch.load(PATH))
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        accuracies.append(round(eval(net, testloader, device, None)[0].item() * 100, 2))

    plt.plot(RATIOS, accuracies, color='red', marker='o')
    dict = {'vgg11': 'VGG11', 'resnet20': 'ResNet20'}
    plt.title(f'{dict[args.model_name.lower()]} Channel Expansion on CIFAR100', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    for i, accuracy in enumerate(accuracies):
        if i + 1 == len(accuracies) or i + 1 == len(accuracies) - 1 or i + 1 == len(accuracies) - 2 or \
                i + 1 == len(accuracies) - 3:
            plt.annotate(accuracy, (RATIOS[i], accuracies[i]), textcoords="offset points", xytext=(-13, -20))
        else:
            plt.annotate(accuracy, (RATIOS[i], accuracies[i]), textcoords="offset points", xytext=(-13, 20))
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + f'/{args.model_name.lower()}_channel_expansion.png')
