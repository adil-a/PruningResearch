import os
import matplotlib.pyplot as plt
from typing import List, Union

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

from Utils.Network_Retrieval import get_network
from OverparameterizationVerification import val

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def multiplier(cfg: List[Union[int, str]], ratio: float):
    for i in range(len(cfg)):
        if isinstance(cfg[i], int):
            cfg[i] = int(ratio * cfg[i])


def main():
    BATCH_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    torch.manual_seed(0)
    RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    accuracies = []
    for ratio in RATIOS:
        current_cfg = defaultcfg[11].copy()
        multiplier(current_cfg, ratio)
        file_to_open = f'vgg11_{ratio}x_best.pt'
        PATH = os.getcwd() + f'/Models/SavedModels/{file_to_open}'
        net = get_network('vgg11', 'cifar100', current_cfg)
        net.load_state_dict(torch.load(PATH))
        net.to(device)
        net.eval()
        accuracies.append(round(val(net, testloader, device).item() * 100, 2))

    plt.plot(RATIOS, accuracies, color='red', marker='o')
    plt.title('VGG11 Channel Expansion on CIFAR100', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    for i, accuracy in enumerate(accuracies):
        if i + 1 == len(accuracies) or i + 1 == len(accuracies) - 1:
            plt.annotate(accuracy, (RATIOS[i], accuracies[i]), textcoords="offset points", xytext=(-13, -20))
        else:
            plt.annotate(accuracy, (RATIOS[i], accuracies[i]), textcoords="offset points", xytext=(-13, 20))
    plt.savefig(os.getcwd() + '/vgg11_channel_expansion.png')


if __name__ == '__main__':
    main()
