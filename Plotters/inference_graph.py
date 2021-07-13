import os
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch

from Utils.config import PRIVATE_PATH, BATCH_SIZE, SEED
from Utils.network_utils import get_network, multiplier, get_test_loader
from OverparameterizationVerification import val

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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
    all_files = os.listdir(PRIVATE_PATH + '/Models/SavedModels/expansion_ratio_inference/')
    for ratio in ratios:
        for file in all_files:
            if str(ratio) in file and 'best' in file:
                lst.append(file)
    return lst


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    testloader = get_test_loader(BATCH_SIZE)
    RATIOS = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    accuracies = []
    for ratio in RATIOS:
        current_cfg = defaultcfg[11].copy()
        multiplier(current_cfg, ratio)
        file_to_open = f'vgg11_{ratio}x_best.pt'
        PATH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/{file_to_open}'
        net = get_network('vgg11', 'cifar100', current_cfg)
        state_dict = load_for_non_parallel(torch.load(PATH))
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        accuracies.append(round(val(net, testloader, device, None)[0].item() * 100, 2))

    plt.plot(RATIOS, accuracies, color='red', marker='o')
    plt.title('VGG11 Channel Expansion on CIFAR100', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    for i, accuracy in enumerate(accuracies):
        if i + 1 == len(accuracies) or i + 1 == len(accuracies) - 1 or i + 1 == len(accuracies) - 2 or \
                i + 1 == len(accuracies) - 3:
            plt.annotate(accuracy, (RATIOS[i], accuracies[i]), textcoords="offset points", xytext=(-13, -20))
        else:
            plt.annotate(accuracy, (RATIOS[i], accuracies[i]), textcoords="offset points", xytext=(-13, 20))
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '/vgg11_channel_expansion.png')


if __name__ == '__main__':
    main()
