from OverparameterizationVerification import val
from Utils import pruning_utils, network_utils
from config import PRIVATE_PATH, BATCH_SIZE

import os
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.nn.utils import prune

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def plot_num_of_parameters(ratios, target_size, device):
    before_pruning = []
    after_pruning = []
    for ratio in ratios:
        curr_cfg = defaultcfg[11].copy()
        network_utils.multiplier(curr_cfg, ratio)
        net = network_utils.get_network('vgg11', 'cifar100', curr_cfg)
        net.to(device)
        if ratio == 1.0:
            num_of_params = round(pruning_utils.measure_number_of_parameters(net) / 1000000, 2)
            before_pruning.append(num_of_params)
            after_pruning.append(num_of_params)
        else:
            num_of_params = pruning_utils.measure_number_of_parameters(net)
            before_pruning.append(round(num_of_params / 1000000, 2))
            final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
            path = PRIVATE_PATH + f'/Models/SavedModels/Finetune/vgg11_{ratio}x_finetune_{final_model_number}_best.pt'
            for module_name, module in net.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    prune.identity(module, "weight")
                    prune.identity(module, "bias")
            net.load_state_dict(torch.load(path))
            after_pruning.append(round((num_of_params - pruning_utils.measure_global_sparsity(net)[0]) / 1000000, 2))
    plt.plot(ratios, before_pruning, color='red', marker='o', label='Before')
    plt.plot(ratios, after_pruning, color='blue', marker='x', label='After')
    plt.title('Number of Parameters in VGG11 Model Before/After Pruning', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Number of Parameters (Millions)', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    for i, size in enumerate(before_pruning):
        if i + 1 != len(before_pruning):
            plt.annotate(size, (ratios[i], before_pruning[i]), textcoords="offset points", xytext=(-13, 20))
        else:
            plt.annotate(size, (ratios[i], before_pruning[i]), textcoords="offset points", xytext=(-18, -27))
    for i, size in enumerate(after_pruning):
        if i != 0:
            plt.annotate(size, (ratios[i], after_pruning[i]), textcoords="offset points", xytext=(-13, 10))
    plt.savefig(os.getcwd() + '/vgg11_num_of_parameters.png')


def pruning_accuracies(ratios, target_size, test_loader, device):
    folder_names = ['expansion_ratio_inference', 'Finetune', 'LR_Rewind', 'Reinitialize']
    labels = ['Unpruned', 'Finetune', 'LR Rewind', 'Reinitialize']
    temp_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    colors = ['red', 'green', 'blue', 'black']
    markers = ['o', 'x', 's', '*']
    for folder_name in folder_names:
        accuracies = []
        path = PRIVATE_PATH + f'/Models/SavedModels/{folder_name}/'
        for ratio in ratios:
            print(f'{ratio}, {folder_name}')
            curr_cfg = defaultcfg[11].copy()
            network_utils.multiplier(curr_cfg, ratio)
            net = network_utils.get_network('vgg11', 'cifar100', curr_cfg)
            if folder_name == 'expansion_ratio_inference':
                net.load_state_dict(torch.load(path + f'vgg11_{ratio}x_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(val(net, test_loader, device, None)[0].item() * 100, 2))
            else:
                if ratio != 1.0:
                    num_of_params = pruning_utils.measure_number_of_parameters(net)
                    final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                    for module_name, module in net.named_modules():
                        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                            prune.identity(module, "weight")
                            prune.identity(module, "bias")
                    net.load_state_dict(torch.load(path +
                                                   f'vgg11_{ratio}x_{folder_name.lower()}_{final_model_number}_best.pt'))
                    pruning_utils.remove_parameters(net)
                    net.to(device)
                    net.eval()
                    accuracies.append(round(val(net, test_loader, device, None)[0].item() * 100, 2))
        idx = folder_names.index(folder_name)
        if folder_name == 'expansion_ratio_inference':
            plt.plot(ratios, accuracies, color=colors[idx], marker=markers[idx], label=labels[idx])
        else:
            plt.plot(temp_ratios, accuracies, color=colors[idx], marker=markers[idx], label=labels[idx])
    plt.title('VGG11 Accuracies w/ Different Magnitude Pruning Techniques', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.getcwd() + '/vgg11_pruning_accuracy.png')


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SIZE = pruning_utils.measure_number_of_parameters(network_utils.get_network('vgg11', 'cifar100',
                                                                                       defaultcfg[11].copy()))
    RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # plot_num_of_parameters(RATIOS, TARGET_SIZE, device)
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    pruning_accuracies(RATIOS, TARGET_SIZE, testloader, device)


if __name__ == '__main__':
    main()
