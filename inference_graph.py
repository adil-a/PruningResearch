import os

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


def main():
    BATCH_SIZE = 128
    torch.manual_seed(0)
    RATIOS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    current_ratio = RATIOS[1]
    current_cfg = defaultcfg[11]
    file_to_open = f'vgg11_{current_ratio}x_best.pt'
    PATH = os.getcwd() + f'/Models/SavedModels/{file_to_open}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    for i in range(len(current_cfg)):
        if isinstance(current_cfg[i], int):
            current_cfg[i] = int(current_ratio * current_cfg[i])

    net = get_network('vgg11', 'cifar100', current_cfg)
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    net.eval()

    print(val(net, testloader, device).item())


if __name__ == '__main__':
    main()
