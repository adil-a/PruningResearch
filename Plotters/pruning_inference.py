from Utils import pruning_utils, network_utils
from Utils.config import PRIVATE_PATH, BATCH_SIZE, SEED, TARGET_SIZE, defaultcfg

import os
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.nn.utils import prune


def plot_num_of_parameters(ratios, target_size, device):
    before_pruning = []
    after_pruning = []
    for ratio in ratios:
        curr_cfg = defaultcfg[11].copy()
        network_utils.multiplier(curr_cfg, ratio)
        net = network_utils.get_network('vgg11', 'cifar100', curr_cfg)
        net.to(device)
        num_of_params = pruning_utils.measure_number_of_parameters(net)
        before_pruning.append(round(num_of_params / 1000000, 1))
        final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
        path = PRIVATE_PATH + f'/Models/SavedModels/Finetune/vgg11_{ratio}x_finetune_{final_model_number}_best.pt'
        for module_name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.identity(module, "weight")
                prune.identity(module, "bias")
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


def IMP_pruning_accuracies(ratios, target_size, test_loader, device):
    folder_names = ['expansion_ratio_inference', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
    labels = ['Unpruned', 'Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
    colors = ['red', 'green', 'blue', 'black', 'purple']
    markers = ['o', 'x', 's', '*', '^']
    for folder_name in folder_names:
        accuracies = []
        temp_ratios = []
        if folder_name == 'SNIP' or folder_name == 'SynFlow':
            path = PRIVATE_PATH + f'/Models/SavedModels/{folder_name}/old/'  # TODO fix this when training is done
        else:
            path = PRIVATE_PATH + f'/Models/SavedModels/{folder_name}/'
        for ratio in ratios:
            print(f'{ratio}, {folder_name}')
            curr_cfg = defaultcfg[11].copy()
            network_utils.multiplier(curr_cfg, ratio)
            if folder_name == 'SNIP' or folder_name == 'SynFlow':
                net = network_utils.get_network('vgg11', 'cifar100', curr_cfg, imp=False)
            else:
                net = network_utils.get_network('vgg11', 'cifar100', curr_cfg)
            if folder_name == 'expansion_ratio_inference' or folder_name == 'SNIP' or folder_name == 'SynFlow':
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
                for module_name, module in net.named_modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        prune.identity(module, "weight")
                        prune.identity(module, "bias")
                net.load_state_dict(torch.load(path +
                                               f'vgg11_{ratio}x_{folder_name.lower()}_{final_model_number}_best.pt'))
                pruning_utils.remove_parameters(net)
                net.to(device)
                net.eval()
                accuracies.append(round(network_utils.eval(net, test_loader, device, None)[0].item() * 100, 2))
                temp_ratios.append(ratio)
        idx = folder_names.index(folder_name)
        plt.plot(ratios, accuracies, color=colors[idx], marker=markers[idx], label=labels[idx])
    plt.title('VGG11 Accuracies w/ Different Magnitude Pruning Techniques', fontsize=14)
    plt.xlabel('Expansion Ratio', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '')) + '/vgg11_pruning_accuracy.png')


def weights_per_layers_helper(model, imp):
    x = []
    y = []
    counter = 1

    nonfc_layer_num_to_mask = {}
    fc_layer_num_to_mask = {}

    if imp:
        for buffer_name, buffer in model.named_buffers():
            layer_number = [int(i) for i in buffer_name.split('.') if i.isdigit()][0]
            if 'weight_mask' in buffer_name or 'bias_mask' in buffer_name:
                remaining_params = buffer.detach().cpu().numpy().sum()
                total_params = buffer.numel()
                if "feature" in buffer_name:
                    if layer_number in nonfc_layer_num_to_mask:
                        nonfc_layer_num_to_mask[layer_number].append((remaining_params, total_params))
                    else:
                        nonfc_layer_num_to_mask[layer_number] = [(remaining_params, total_params)]
                elif "classifier" in buffer_name:
                    if layer_number in fc_layer_num_to_mask:
                        fc_layer_num_to_mask[layer_number].append((remaining_params, total_params))
                    else:
                        fc_layer_num_to_mask[layer_number] = [(remaining_params, total_params)]
    else:
        illegal_num = -1  # so we don't track BN layers
        for buffer_name, buffer in model.named_buffers():
            layer_number = [int(i) for i in buffer_name.split('.') if i.isdigit()][0]
            if ("running_mean" in buffer_name) or ("running_var" in buffer_name) or \
                    ("num_batches_tracked" in buffer_name):
                illegal_num = layer_number
            if 'weight_mask' in buffer_name or 'bias_mask' in buffer_name:
                remaining_params = buffer.detach().cpu().numpy().sum()
                total_params = buffer.numel()
                if "features" in buffer_name and layer_number != illegal_num:
                    if layer_number in nonfc_layer_num_to_mask:
                        nonfc_layer_num_to_mask[layer_number].append((remaining_params, total_params))
                    else:
                        nonfc_layer_num_to_mask[layer_number] = [(remaining_params, total_params)]
                elif "classifier" in buffer_name:
                    if layer_number in fc_layer_num_to_mask:
                        fc_layer_num_to_mask[layer_number].append((remaining_params, total_params))
                    else:
                        fc_layer_num_to_mask[layer_number] = [(remaining_params, total_params)]
    nonfc_keys = list(nonfc_layer_num_to_mask.keys())
    fc_keys = list(fc_layer_num_to_mask.keys())
    nonfc_keys.sort()
    fc_keys.sort()
    for key in nonfc_keys:
        running_total = 0
        running_remaining = 0
        for pair in nonfc_layer_num_to_mask[key]:
            running_remaining += pair[0]
            running_total += pair[1]
        x.append(counter)
        y.append(round((running_remaining / running_total) * 100, 1))
        counter += 1
    for key in fc_keys:
        running_total = 0
        running_remaining = 0
        for pair in fc_layer_num_to_mask[key]:
            running_remaining += pair[0]
            running_total += pair[1]
        x.append(counter)
        y.append(round((running_remaining / running_total) * 100, 1))
        counter += 1
    return x, y


def weights_per_layers(ratios, device, target_size):
    folder_names = ['Finetune', 'Reinitialize', 'SNIP', 'SynFlow']
    colors = ['red', 'green', 'blue', 'black', 'purple', 'mediumslateblue', 'maroon']
    markers = ['o', 'x', 's', '*', '^', 'D', 'p']
    fig, axs = plt.subplots(len(folder_names), 1, figsize=(15, 15))
    for i in range(len(folder_names)):
        if folder_names[i] == 'SNIP' or folder_names[i] == 'SynFlow':
            path = PRIVATE_PATH + f'/Models/SavedModels/{folder_names[i]}/'  # TODO fix this when training is done
        else:
            path = PRIVATE_PATH + f'/Models/SavedModels/{folder_names[i]}/'
        for j in range(len(ratios)):
            curr_cfg = defaultcfg[11].copy()
            network_utils.multiplier(curr_cfg, ratios[j])
            if folder_names[i] == 'SNIP' or folder_names[i] == 'SynFlow':
                net = network_utils.get_network('vgg11', 'cifar100', curr_cfg, imp=False)
            else:
                net = network_utils.get_network('vgg11', 'cifar100', curr_cfg)
            if folder_names[i] == 'SNIP' or folder_names[i] == 'SynFlow':
                net.load_state_dict(torch.load(path + f'vgg11_{ratios[j]}x_best.pt'))
                net.to(device)
                net.eval()
                x, y = weights_per_layers_helper(net, False)
            else:
                num_of_params = pruning_utils.measure_number_of_parameters(net)
                if folder_names[i] == 'Reinitialize':
                    final_model_number = 1
                else:
                    final_model_number = pruning_utils.get_finetune_iterations(target_size, num_of_params, 0.2)
                for module_name, module in net.named_modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        prune.identity(module, "weight")
                        prune.identity(module, "bias")
                net.load_state_dict(torch.load(path +
                                               f'vgg11_{ratios[j]}x_{folder_names[i].lower()}_'
                                               f'{final_model_number}_best.pt'))
                net.to(device)
                net.eval()
                x, y = weights_per_layers_helper(net, True)
            axs[i].plot(x, y, color=colors[j], marker=markers[j], label=str(ratios[j]))
            axs[i].set_title(folder_names[i])
            axs[i].set_xlabel('Layer Number')
            axs[i].set_ylabel('Percentage Weights Remaining')
            axs[i].grid(True)
            axs[i].legend(loc="upper left")
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
    if args.graph == 'num_of_params':
        plot_num_of_parameters(RATIOS, TARGET_SIZE, device)
    elif args.graph == 'accuracies':
        IMP_pruning_accuracies(RATIOS, TARGET_SIZE, testloader, device)
    elif args.graph == 'weights_per_layer':
        weights_per_layers(RATIOS, device, TARGET_SIZE)
