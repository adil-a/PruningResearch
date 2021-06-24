from OverparameterizationVerification import val
from Utils import pruning_utils, network_utils
from config import PRIVATE_PATH, BATCH_SIZE

import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    RATIOS = [1.0, 1.5, 2.0]
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
    TARGET_SIZE = pruning_utils.measure_number_of_parameters(network_utils.get_network('vgg11', 'cifar100',
                                                                                       defaultcfg[11].copy()))
    for ratio in RATIOS:
        testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
        accuracies = []
        sparsities = []
        current_cfg = defaultcfg[11].copy()
        network_utils.multiplier(current_cfg, ratio)
        net_size = pruning_utils.measure_number_of_parameters(network_utils.get_network('vgg11', 'cifar100',
                                                                                        current_cfg))
        num_of_files = pruning_utils.get_finetune_iterations(TARGET_SIZE, net_size, 0.2)
        dir_path = PRIVATE_PATH + '/Models/SavedModels/Finetune/'
        for i in range(num_of_files):
            net = network_utils.get_network('vgg11', 'cifar100', current_cfg)
            file_path = dir_path + f'vgg11_{ratio}x_finetune_{i + 1}_best.pt'
            net.load_state_dict(torch.load(file_path))
            net.train()
            sparsities.append(pruning_utils.measure_global_sparsity(net)[2])
            net.eval()
            accuracies.append(val(net, testloader, device, None)[0])


if __name__ == '__main__':
    main()
