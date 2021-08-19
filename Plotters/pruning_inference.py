from Utils import pruning_utils, network_utils
from Utils.config import PRIVATE_PATH, BATCH_SIZE, SEED, VGG_TARGET_SIZE, defaultcfg_vgg
from Layers import layers

import os
import matplotlib.pyplot as plt
from math import log10

import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def plot_num_of_parameters(ratios, target_size, device):
    before_pruning = []
    after_pruning = []
    for ratio in ratios:
        curr_cfg = defaultcfg_vgg[11].copy()
        network_utils.multiplier(curr_cfg, ratio)
        net = network_utils.get_network('vgg11', 'cifar100', curr_cfg, imp=False)
        net.to(device)
        num_of_params = pruning_utils.measure_number_of_parameters(net)
        before_pruning.append(round(num_of_params / 1000000, 1))
        final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune/vgg11_{ratio}x_finetune_{final_model_number}_best.pt'
        net.load_state_dict(torch.load(path))
        after_pruning.append(
            round((num_of_params - pruning_utils.measure_global_sparsity(net)[0]) / 1000000, 1))
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
        plt.annotate(size, (ratios[i], after_pruning[i]), textcoords="offset points", xytext=(-13, 10))
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + '/vgg11_num_of_parameters.png')


def IMP_other_accuracies(ratios, target_size, test_loader, device, folder_names, labels, colors, markers, title,
                         plot_path):
    plt.figure(figsize=(10, 10))
    for folder_name in folder_names:
        accuracies = []
        temp = ''
        if 'Finetune_Mask_Mix' in folder_name:
            folder_name, temp = folder_name.split(sep='/')
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/{folder_name}/'
        for ratio in ratios:
            if temp != '':
                print(f'{ratio}, {folder_name + "/" + temp}')
            else:
                print(f'{ratio}, {folder_name}')
            curr_cfg = defaultcfg_vgg[11].copy()
            network_utils.multiplier(curr_cfg, ratio)
            net = network_utils.get_network('vgg11', 'cifar100', curr_cfg, imp=False)
            if folder_name == 'SynFlow' or folder_name == 'SNIP':
                net.load_state_dict(torch.load(path + f'vgg11_{ratio}x_{folder_name}_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
            elif folder_name == 'Finetune':
                num_of_params = pruning_utils.measure_number_of_parameters(net)
                final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                net.load_state_dict(torch.load(path +
                                               f'vgg11_{ratio}x_{folder_name.lower()}_{final_model_number}_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
            elif folder_name == 'Finetune_Singleshot':
                net.load_state_dict(torch.load(path + f'vgg11_{ratio}x_finetune_singleshot_1_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
            else:
                if temp == 'SNIP' and ratio == 4.0:
                    accuracies.append(None)
                    continue
                net.load_state_dict(torch.load(path + f'vgg11_{ratio}x_{temp}_MaskMix_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
        if temp != '':
            idx = folder_names.index(folder_name + '/' + temp)
        else:
            idx = folder_names.index(folder_name)
        plt.plot(ratios, accuracies, color=colors[idx], marker=markers[idx], label=labels[idx])
    plt.title(title, fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + f'/{plot_path}')


def IMP_pruning_accuracies(ratios, target_size, test_loader, device):
    folder_names = ['expansion_ratio_inference', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
    labels = ['Unpruned', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
    colors = ['red', 'green', 'maroon', 'black', 'purple']
    markers = ['o', 'x', 'D', '*', '^']
    for folder_name in folder_names:
        accuracies = []
        temp_ratios = []
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/{folder_name}/'
        for ratio in ratios:
            print(f'{ratio}, {folder_name}')
            curr_cfg = defaultcfg_vgg[11].copy()
            network_utils.multiplier(curr_cfg, ratio)
            net = network_utils.get_network('vgg11', 'cifar100', curr_cfg, imp=False)
            if folder_name == 'expansion_ratio_inference' or folder_name == 'SNIP' or folder_name == 'SynFlow':
                if folder_name != 'expansion_ratio_inference':
                    net.load_state_dict(torch.load(path + f'vgg11_{ratio}x_{folder_name}_best.pt'))
                else:
                    net.load_state_dict(torch.load(path + f'vgg11_{ratio}x_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
                temp_ratios.append(ratio)
            else:
                num_of_params = pruning_utils.measure_number_of_parameters(net)
                if folder_name == 'Reinitialize':
                    final_model_number = 1
                else:
                    final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                net.load_state_dict(torch.load(path +
                                               f'vgg11_{ratio}x_{folder_name.lower()}_{final_model_number}_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
                temp_ratios.append(ratio)
        idx = folder_names.index(folder_name)
        plt.plot(ratios, accuracies, color=colors[idx], marker=markers[idx], label=labels[idx])
    plt.title('VGG11 Accuracies w/ Different Pruning Techniques', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + '/vgg11_pruning_accuracy.png')


def neuron_calculator(module):
    if isinstance(module, layers.Conv2d):
        return module.weight.size()[0]  # number of out channels
    elif isinstance(module, layers.Linear):
        return module.weight.size()[0] + 1  # + 1 for bias neuron


def weights_per_layers_helper(model):
    x_conv = []
    y_conv = []
    x_linear = []
    y_linear = []
    counter_conv = 1
    counter_linear = 1
    for module_name, module in model.named_modules():
        if isinstance(module, (layers.Conv2d, layers.Linear)):
            num_neurons = neuron_calculator(module)
            remaining_params = module.weight_mask.flatten().sum().item() + module.bias_mask.flatten().sum().item()
            if isinstance(module, layers.Conv2d):
                x_conv.append(counter_conv)
                y_conv.append(log10(remaining_params / num_neurons))
                counter_conv += 1
            else:
                x_linear.append(counter_linear)
                y_linear.append(log10(remaining_params / num_neurons))
                counter_linear += 1
    return x_conv, y_conv, x_linear, y_linear


def weights_per_layers(ratios, device, target_size):
    # folder_names = ['Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
    folder_names = ['Finetune', 'SNIP', 'SynFlow']
    colors = ['red', 'green', 'blue', 'black', 'purple', 'mediumslateblue', 'maroon']
    markers = ['o', 'x', 's', '*', '^', 'D', 'p']
    fig, axs = plt.subplots(len(folder_names), 2, figsize=(15, 15))
    for i in range(len(folder_names)):
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/{folder_names[i]}/'
        for j in range(len(ratios)):
            curr_cfg = defaultcfg_vgg[11].copy()
            network_utils.multiplier(curr_cfg, ratios[j])
            net = network_utils.get_network('vgg11', 'cifar100', curr_cfg, imp=False)
            if folder_names[i] == 'SNIP' or folder_names[i] == 'SynFlow':
                net.load_state_dict(torch.load(path + f'vgg11_{ratios[j]}x_{folder_names[i]}_best.pt'))
                net.to(device)
                net.eval()
                x_conv, y_conv, x_linear, y_linear = weights_per_layers_helper(net)
            else:
                num_of_params = pruning_utils.measure_number_of_parameters(net)
                if folder_names[i] == 'Reinitialize':
                    final_model_number = 1
                else:
                    final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                net.load_state_dict(torch.load(path +
                                               f'vgg11_{ratios[j]}x_{folder_names[i].lower()}_'
                                               f'{final_model_number}_best.pt'))
                net.to(device)
                net.eval()
                x_conv, y_conv, x_linear, y_linear = weights_per_layers_helper(net)
            axs[i][0].plot(x_conv, y_conv, color=colors[j], marker=markers[j], label=str(ratios[j]))
            axs[i][1].plot(x_linear, y_linear, color=colors[j], marker=markers[j], label=str(ratios[j]))
            axs[i][0].set_title(folder_names[i])
            axs[i][1].set_title(folder_names[i])
            axs[i][0].set_xlabel('Layer Number')
            axs[i][1].set_xlabel('Layer Number')
            axs[i][0].set_ylabel('Remaining Params / Channels')
            axs[i][1].set_ylabel('Remaining Params / Neurons')
            axs[i][0].set_ylim([1, 3.5])
            axs[i][1].set_ylim([1, 3.5])
            axs[i][0].grid(True)
            axs[i][1].grid(True)
            axs[i][0].legend(loc="upper left")
            axs[i][1].legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + '/vgg11_weights_per_layer.png')


def main(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    metadata = {'mask_mix': (['Finetune', 'Finetune_Mask_Mix', 'SynFlow'],
                             ['Finetune', 'Finetune w/ SynFlow Mask', 'SynFlow'],
                             ['green', 'black', 'purple'],
                             ['o', 'x', '*'],
                             'VGG11 Accuracies w/ Mask Mixing',
                             'vgg11_pruning_mask_mix_accuracy.png'),
                'singleshot_imp': (['Finetune', 'Finetune_Singleshot', 'Finetune_Mask_Mix/SNIP',
                                    'Finetune_Mask_Mix/SynFlow', 'SynFlow', 'SNIP'],
                                   ['Finetune', 'Finetune w/ Singleshot Magnitude Pruning', 'Finetune w/ SNIP Mask',
                                    'Finetune w/ SynFlow Mask', 'SynFlow', 'SNIP'],
                                   ['red', 'blue', 'purple', 'black', 'green', 'maroon'],
                                   ['o', 'x', '*', 'D', '^', 'p'],
                                   'VGG11 Accuracies w/ Different Singleshot Masks',
                                   'vgg11_singleshot.png')
                }
    if args.graph == 'num_of_params':
        plot_num_of_parameters(RATIOS, VGG_TARGET_SIZE, device)
    elif args.graph == 'pruned_accuracies':
        IMP_pruning_accuracies(RATIOS, VGG_TARGET_SIZE, testloader, device)
    elif args.graph == 'weights_per_layer':
        weights_per_layers(RATIOS, device, VGG_TARGET_SIZE)
    elif args.graph == 'mask_mix' or args.graph == 'singleshot_imp':
        folder_names, labels, colors, markers, title, path = metadata[args.graph]
        IMP_other_accuracies(RATIOS, VGG_TARGET_SIZE, testloader, device, folder_names, labels, colors, markers, title,
                             path)
