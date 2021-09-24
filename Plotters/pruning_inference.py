from Utils import pruning_utils, network_utils
from Utils.config import PRIVATE_PATH, BATCH_SIZE, SEED, VGG_TARGET_SIZE, defaultcfg_vgg, defaultcfg_resnet_imagenet, \
    defaultcfg_resnet_cifar, RESNET_CIFAR_TARGET_SIZE
from Layers import layers

import os
import matplotlib.pyplot as plt
from math import log10

import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def plot_num_of_parameters(ratios, target_size, device, args):
    before_pruning = []
    after_pruning = []
    for ratio in ratios:
        if 'vgg' in args.model_name.lower():
            curr_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
        else:
            if args.dataset.lower() == 'imagenet':
                curr_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
            else:
                curr_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
        network_utils.multiplier(curr_cfg, ratio)
        net = network_utils.get_network(args.model_name, args.dataset, curr_cfg)
        net.to(device)
        num_of_params = pruning_utils.measure_number_of_parameters(net)
        before_pruning.append(round(num_of_params / 1000000, 1))
        if args.model_name.lower() == 'resnet20' and ratio == 3.0:
            after_pruning.append(round(num_of_params / 1000000, 1))
            continue
        final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
        if 'vgg' in args.model_name.lower():
            path = PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune/{args.model_name.lower()}_{ratio}x_finetune_' \
                                  f'{final_model_number}_best.pt'
        elif 'resnet' in args.model_name.lower():
            path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/Finetune/{args.model_name.lower()}_{ratio}x_finetune_' \
                                  f'{final_model_number}_best.pt'
        net.load_state_dict(torch.load(path))
        after_pruning.append(
            round((num_of_params - pruning_utils.measure_global_sparsity(net)[0]) / 1000000, 1))
    plt.plot(ratios, before_pruning, color='red', marker='o', label='Before')
    plt.plot(ratios, after_pruning, color='blue', marker='x', label='After')
    dict = {'vgg11': 'VGG11', 'resnet20': 'ResNet20'}
    plt.title(f'Number of Parameters in {dict[args.model_name.lower()]} Model Before/After Pruning', fontsize=14)
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
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + f'/{args.model_name.lower()}_'
                                                                               f'num_of_parameters.png')


def IMP_other_accuracies(ratios, target_size, test_loader, device, folder_names, labels, colors, markers, title,
                         plot_path, args):
    plt.figure(figsize=(10, 10))
    for folder_name in folder_names:
        accuracies = []
        temp = ''
        if 'Finetune_Mask_Mix' in folder_name:
            folder_name, temp = folder_name.split(sep='/')
        if 'vgg' in args.model_name.lower():
            path = PRIVATE_PATH + f'/Models/SavedModels/VGG/{folder_name}/'
        elif 'resnet' in args.model_name.lower():
            path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/{folder_name}/'
        for ratio in ratios:
            if ratio == 3.0 or ((ratio == 20.0 or ratio == 15.0) and folder_name == 'Finetune_Mask_Mix'
                                and temp == 'SynFlow') or (ratio == 20.0 and temp == 'SNIP'):
                accuracies.append(None)
                continue
            if temp != '':
                print(f'{ratio}, {folder_name + "/" + temp}')
            else:
                print(f'{ratio}, {folder_name}')
            if 'vgg' in args.model_name.lower():
                curr_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
            else:
                if args.dataset.lower() == 'imagenet':
                    curr_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
                else:
                    curr_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
            network_utils.multiplier(curr_cfg, ratio)
            net = network_utils.get_network(args.model_name, args.dataset, curr_cfg)
            if folder_name == 'SynFlow' or folder_name == 'SNIP':
                net.load_state_dict(torch.load(path + f'{args.model_name.lower()}_{ratio}x_{folder_name}_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
            elif folder_name == 'Finetune':
                num_of_params = pruning_utils.measure_number_of_parameters(net)
                final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                net.load_state_dict(torch.load(path +
                                               f'{args.model_name.lower()}_{ratio}x_{folder_name.lower()}_'
                                               f'{final_model_number}_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
            elif folder_name == 'Finetune_Singleshot':
                net.load_state_dict(torch.load(path + f'{args.model_name.lower()}_{ratio}x_finetune_'
                                                      f'singleshot_1_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
            else:
                if temp == 'SNIP' and ratio == 4.0:
                    accuracies.append(None)
                    continue
                net.load_state_dict(torch.load(path + f'{args.model_name.lower()}_{ratio}x_{temp}_MaskMix_best.pt'))
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


def IMP_pruning_accuracies(ratios, target_size, test_loader, device, args):
    if 'vgg' in args.model_name.lower():
        folder_names = ['expansion_ratio_inference', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
        labels = ['Unpruned', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
        colors = ['red', 'green', 'maroon', 'black', 'purple']
        markers = ['o', 'x', 'D', '*', '^']
    elif 'resnet' in args.model_name.lower():
        # folder_names = ['expansion_ratio_inference', 'Finetune', 'Reinitialize', 'SynFlow']
        # labels = ['Unpruned', 'Finetune', 'Reinitialize', 'SynFlow']
        # colors = ['red', 'green', 'maroon', 'purple']
        # markers = ['o', 'x', 'D', '^']
        folder_names = ['expansion_ratio_inference', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
        labels = ['Unpruned', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
        colors = ['red', 'green', 'maroon', 'black', 'purple']
        markers = ['o', 'x', 'D', '*', '^']
    for folder_name in folder_names:
        accuracies = []
        temp_ratios = []
        if 'vgg' in args.model_name.lower():
            path = PRIVATE_PATH + f'/Models/SavedModels/VGG/{folder_name}/'
        elif 'resnet' in args.model_name.lower():
            path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/{folder_name}/'
        for ratio in ratios:
            print(f'{ratio}, {folder_name}')
            if 'vgg' in args.model_name.lower():
                curr_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
            else:
                if args.dataset.lower() == 'imagenet':
                    curr_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
                else:
                    curr_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
            network_utils.multiplier(curr_cfg, ratio)
            net = network_utils.get_network(args.model_name, args.dataset, curr_cfg)
            if folder_name == 'expansion_ratio_inference' or folder_name == 'SNIP' or folder_name == 'SynFlow':
                if (ratio == 3.0) and args.model_name.lower() == 'resnet20' and (folder_name == 'SynFlow' or
                                                                                 folder_name == 'SNIP'):
                    temp_ratios.append(ratio)
                    accuracies.append(None)
                    continue
                if ratio == 20.0 and args.model_name.lower() == 'resnet20' and folder_name == 'SNIP':
                    temp_ratios.append(ratio)
                    accuracies.append(None)
                    continue
                if folder_name != 'expansion_ratio_inference':
                    net.load_state_dict(torch.load(path + f'{args.model_name.lower()}_{ratio}x_{folder_name}_best.pt'))
                else:
                    net.load_state_dict(torch.load(path + f'{args.model_name.lower()}_{ratio}x_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
                temp_ratios.append(ratio)
            else:
                if ratio == 3.0 and args.model_name.lower() == 'resnet20':
                    accuracies.append(None)
                    temp_ratios.append(ratio)
                    continue
                num_of_params = pruning_utils.measure_number_of_parameters(net)
                if folder_name == 'Reinitialize':
                    final_model_number = 1
                else:
                    final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                if folder_name == 'Reinitialize':
                    net.load_state_dict(torch.load(path +
                                                   f'{args.model_name.lower()}_{ratio}x_{folder_name.lower()}'
                                                   f'_weight_rewind_first_epoch_{final_model_number}_best.pt'))
                else:
                    net.load_state_dict(torch.load(path +
                                                   f'{args.model_name.lower()}_{ratio}x_{folder_name.lower()}_'
                                                   f'{final_model_number}_best.pt'))
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
                temp_ratios.append(ratio)
        idx = folder_names.index(folder_name)
        plt.plot(ratios, accuracies, color=colors[idx], marker=markers[idx], label=labels[idx])
    dict = {'vgg11': 'VGG11', 'resnet20': 'ResNet20'}
    plt.title(f'{dict[args.model_name.lower()]} Accuracies w/ Different Pruning Techniques', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + f'/{args.model_name.lower()}_'
                                                                               f'pruning_accuracy.png')


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


def reinit_diff_epochs(ratios, test_loader, device, args):
    epoch_data = {0: ('first', []), 1: ('fifth', []), 2: ('tenth', [])}
    if 'vgg' in args.model_name.lower():
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/Reinitialize'
    elif 'resnet' in args.model_name.lower():
        path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/Reinitialize'
    colours = ['green', 'black', 'purple']
    markers = ['o', 'x', '*']
    labels = ['Reinitialize to 1st epoch', 'Reinitialize to 5th epoch', 'Reinitialize to 10th epoch']

    for ratio in ratios:
        for i in range(3):
            if 'resnet' in args.model_name.lower() and ratio == 3.0:
                epoch_data[i][1].append(None)
                continue
            if 'vgg' in args.model_name.lower():
                curr_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
            else:
                if args.dataset.lower() == 'imagenet':
                    curr_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
                else:
                    curr_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
            network_utils.multiplier(curr_cfg, ratio)
            net = network_utils.get_network(args.model_name, args.dataset, curr_cfg)
            net.load_state_dict(torch.load(path + f'/{args.model_name.lower()}_{ratio}x_reinitialize_weight_rewind_'
                                                  f'{epoch_data[i][0]}_epoch_1_best.pt'))
            net.to(device)
            net.eval()
            epoch_data[i][1].append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))

    for i in range(3):
        plt.plot(ratios, epoch_data[i][1], color=f'{colours[i]}', marker=f'{markers[i]}', label=f'{labels[i]}')

    dict = {'vgg11': 'VGG11', 'resnet20': 'ResNet20'}
    plt.title(f'Accuracies after rewinding weights to different epochs for {dict[args.model_name.lower()]} '
              f'{args.dataset.upper()}', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    # for i in range(3):
    #     for j, accuracy in enumerate(epoch_data[i][1]):
    #         plt.annotate(accuracy, (ratios[i], epoch_data[i][1][j]))
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + f'/{args.model_name.lower()}_'
                                                                               f'weight_rewinding.png')


def rewind_masks(ratios, test_loader, device, args):
    epoch_data = {0: (5, {"SynFlow": [], "SNIP": []}), 1: (10, {"SynFlow": [], "SNIP": []})}
    if 'vgg' in args.model_name.lower():
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix'
    elif 'resnet' in args.model_name.lower():
        path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/Finetune_Mask_Mix'
    colours = ['black', 'purple', 'blue', 'green']
    markers = ['x', '*', '+', '^']
    labels = ['Rewind masks to 5th epoch (SynFlow) on final weights',
              'Rewind masks to 10th epoch (SynFlow) on final weights',
              'Rewind masks to 5th epoch (SNIP) on final weights',
              'Rewind masks to 10th epoch (SNIP) on final weights']
    pruning_types = ['SynFlow', 'SNIP']

    for ratio in ratios:
        for i in range(2):
            for pruning_type in pruning_types:
                if 'vgg' in args.model_name.lower() and ratio == 4.0 and i == 1:
                    epoch_data[i][1][pruning_type].append(None)
                    continue
                if 'resnet' in args.model_name.lower() and (ratio == 3.0 or ratio == 20.0):
                    epoch_data[i][1][pruning_type].append(None)
                    continue
                if 'vgg' in args.model_name.lower():
                    curr_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
                else:
                    if args.dataset.lower() == 'imagenet':
                        curr_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
                    else:
                        curr_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
                network_utils.multiplier(curr_cfg, ratio)
                net = network_utils.get_network(args.model_name, args.dataset, curr_cfg)
                net.load_state_dict(torch.load(path + f'/{args.model_name.lower()}_{ratio}x_{pruning_type}_MaskMix_'
                                                      f'masks_rewind_epoch_{epoch_data[i][0]}_best.pt'))
                net.to(device)
                net.eval()
                epoch_data[i][1][pruning_type].append(round(network_utils.eval(net, test_loader,
                                                                               device, None)[0].item() * 100, 2))
    temp_counter = 0
    for i in range(2):
        for pruning_type in pruning_types:
            plt.plot(ratios, epoch_data[i][1][pruning_type], color=f'{colours[temp_counter]}',
                     marker=f'{markers[temp_counter]}', label=f'{labels[temp_counter]}')
            temp_counter += 1

    dict = {'vgg11': 'VGG11', 'resnet20': 'ResNet20'}
    plt.title(f'Accuracies after rewinding masks to different epochs for {dict[args.model_name.lower()]} '
              f'{args.dataset.upper()}', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + f'/{args.model_name.lower()}_'
                                                                               f'mask_rewinding.png')


def rewind_masks_weights(ratios, test_loader, device, args):
    epoch_data = {0: (5, {"Mag": []}), 1: (10, {"Mag": []})}
    if 'vgg' in args.model_name.lower():
        path = PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix'
    elif 'resnet' in args.model_name.lower():
        path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/Finetune_Mask_Mix'
    colours = ['black', 'purple', 'blue', 'green']
    markers = ['x', '*', '+', '^']
    labels = ['Rewind weights & masks to 5th epoch (Mag)',
              'Rewind weights & masks to 10th epoch (Mag)']
    pruning_types = ['Mag']

    for ratio in ratios:
        for i in range(2):
            for pruning_type in pruning_types:
                # if 'vgg' in args.model_name.lower() and ratio == 4.0 and i == 1:
                #     epoch_data[i][1][pruning_type].append(None)
                #     continue
                # if 'resnet' in args.model_name.lower() and (ratio == 3.0 or ratio == 20.0):
                #     epoch_data[i][1][pruning_type].append(None)
                #     continue
                if 'vgg' in args.model_name.lower():
                    curr_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
                else:
                    if args.dataset.lower() == 'imagenet':
                        curr_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
                    else:
                        curr_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
                network_utils.multiplier(curr_cfg, ratio)
                net = network_utils.get_network(args.model_name, args.dataset, curr_cfg)
                net.load_state_dict(torch.load(path + f'/{args.model_name.lower()}_{ratio}x_{pruning_type}_MaskMix_'
                                                      f'masks_rewind_epoch_{epoch_data[i][0]}_best.pt'))
                net.to(device)
                net.eval()
                epoch_data[i][1][pruning_type].append(round(network_utils.eval(net, test_loader,
                                                                               device, None)[0].item() * 100, 2))
    temp_counter = 0
    for i in range(2):
        for pruning_type in pruning_types:
            plt.plot(ratios, epoch_data[i][1][pruning_type], color=f'{colours[temp_counter]}',
                     marker=f'{markers[temp_counter]}', label=f'{labels[temp_counter]}')
            temp_counter += 1

    dict = {'vgg11': 'VGG11', 'resnet20': 'ResNet20'}
    plt.title(f'Accuracies after rewinding masks and weights to different epochs for {dict[args.model_name.lower()]} '
              f'{args.dataset.upper()}', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + f'/{args.model_name.lower()}_'
                                                                               f'weight_mask_rewinding.png')


def main(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RATIOS_VGG = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    RATIOS_RESNET_CIFAR = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    metadata = {'mask_mix': {'vgg11': (['Finetune', 'Finetune_Mask_Mix', 'SynFlow'],
                                       ['Finetune', 'Finetune w/ SynFlow Mask', 'SynFlow'],
                                       ['green', 'black', 'purple'],
                                       ['o', 'x', '*'],
                                       'VGG11 Accuracies w/ Mask Mixing',
                                       'vgg11_pruning_mask_mix_accuracy.png'),
                             'resnet20': (['Finetune', 'Finetune_Mask_Mix', 'SynFlow'],
                                          ['Finetune', 'Finetune w/ SynFlow Mask', 'SynFlow'],
                                          ['green', 'black', 'purple'],
                                          ['o', 'x', '*'],
                                          'ResNet20 Accuracies w/ Mask Mixing',
                                          'resnet20_pruning_mask_mix_accuracy.png')},
                'singleshot_imp': {'vgg11': (['Finetune', 'Finetune_Singleshot', 'Finetune_Mask_Mix/SNIP',
                                              'Finetune_Mask_Mix/SynFlow', 'SynFlow', 'SNIP'],
                                             ['Finetune', 'Finetune w/ Singleshot Magnitude Pruning',
                                              'Finetune w/ SNIP Mask',
                                              'Finetune w/ SynFlow Mask', 'SynFlow', 'SNIP'],
                                             ['red', 'blue', 'purple', 'black', 'green', 'maroon'],
                                             ['o', 'x', '*', 'D', '^', 'p'],
                                             'VGG11 Accuracies w/ Different Singleshot Masks',
                                             'vgg11_singleshot.png'),
                                   'resnet20': (['Finetune', 'Finetune_Singleshot', 'Finetune_Mask_Mix/SynFlow',
                                                 'SynFlow', 'SNIP'],
                                                ['Finetune', 'Finetune w/ Singleshot Magnitude Pruning',
                                                 'Finetune w/ SynFlow Mask', 'SynFlow', 'SNIP'],
                                                ['red', 'blue', 'black', 'green', 'purple'],
                                                ['o', 'x', 'D', '^', '+'],
                                                'ResNet20 Accuracies w/ Different Singleshot Masks',
                                                'resnet20_singleshot.png')}
                }
    if args.graph == 'num_of_params':
        if 'vgg' in args.model_name.lower():
            plot_num_of_parameters(RATIOS_VGG, VGG_TARGET_SIZE, device, args)
        elif 'resnet' in args.model_name.lower() and (args.dataset.lower() == 'cifar100' or args.dataset.lower()
                                                      == 'cifar10'):
            plot_num_of_parameters(RATIOS_RESNET_CIFAR, RESNET_CIFAR_TARGET_SIZE, device, args)
    elif args.graph == 'pruned_accuracies':
        if 'vgg' in args.model_name.lower():
            IMP_pruning_accuracies(RATIOS_VGG, VGG_TARGET_SIZE, testloader, device, args)
        elif 'resnet' in args.model_name.lower() and (args.dataset.lower() == 'cifar100' or args.dataset.lower()
                                                      == 'cifar10'):
            IMP_pruning_accuracies(RATIOS_RESNET_CIFAR, RESNET_CIFAR_TARGET_SIZE, testloader, device, args)
    elif args.graph == 'weights_per_layer':
        weights_per_layers(RATIOS_VGG, device, VGG_TARGET_SIZE)
    elif args.graph == 'mask_mix' or args.graph == 'singleshot_imp':
        folder_names, labels, colors, markers, title, path = metadata[args.graph][args.model_name.lower()]
        if 'vgg' in args.model_name.lower():
            IMP_other_accuracies(RATIOS_VGG, VGG_TARGET_SIZE, testloader, device, folder_names, labels, colors, markers,
                                 title, path, args)
        elif 'resnet' in args.model_name.lower() and (args.dataset.lower() == 'cifar100' or args.dataset.lower()
                                                      == 'cifar10'):
            IMP_other_accuracies(RATIOS_RESNET_CIFAR, RESNET_CIFAR_TARGET_SIZE, testloader, device, folder_names,
                                 labels, colors, markers, title, path, args)
    elif args.graph == 'rewind_epochs':
        if 'vgg' in args.model_name.lower():
            reinit_diff_epochs(RATIOS_VGG, testloader, device, args)
        elif 'resnet' in args.model_name.lower() and (args.dataset.lower() == 'cifar100' or args.dataset.lower()
                                                      == 'cifar10'):
            reinit_diff_epochs(RATIOS_RESNET_CIFAR, testloader, device, args)
    elif args.graph == 'rewind_masks':
        if 'vgg' in args.model_name.lower():
            rewind_masks(RATIOS_VGG, testloader, device, args)
        elif 'resnet' in args.model_name.lower() and (args.dataset.lower() == 'cifar100' or args.dataset.lower()
                                                      == 'cifar10'):
            rewind_masks(RATIOS_RESNET_CIFAR, testloader, device, args)
