import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

from config import PRIVATE_PATH, BATCH_SIZE
from Utils.network_utils import get_network
from OverparameterizationVerification import val
from Utils.network_utils import multiplier

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)

    torch.manual_seed(1)
    RATIOS = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    accuracies = []
    for ratio in RATIOS:
        testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
        current_cfg = defaultcfg[11].copy()
        multiplier(current_cfg, ratio)
        file_to_open = f'vgg11_{ratio}x_best.pt'
        PATH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/{file_to_open}'
        net = get_network('vgg11', 'cifar100', current_cfg)
        net.load_state_dict(torch.load(PATH))
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
    plt.savefig(os.getcwd() + '/vgg11_channel_expansion.png')


if __name__ == '__main__':
    main()
