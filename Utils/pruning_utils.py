# code taken from https://leimao.github.io/blog/PyTorch-Pruning/
import torch
from torch.nn.utils import prune
from math import ceil, log10


def measure_module_sparsity(module):
    num_zeros = 0
    num_elements = 0

    for buffer_name, buffer in module.named_buffers():
        if "weight_mask" in buffer_name or "bias_mask" in buffer_name:
            num_zeros += torch.sum(buffer == 0).item()
            num_elements += buffer.nelement()
    if num_elements != 0:
        sparsity = num_zeros / num_elements
    else:
        sparsity = 0
    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model):
    total_num_zeros = 0
    total_num_elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(module)
            total_num_zeros += module_num_zeros
            total_num_elements += module_num_elements
    if total_num_elements != 0:
        sparsity = total_num_zeros / total_num_elements
    else:
        sparsity = 0
    return total_num_zeros, total_num_elements, sparsity


def measure_number_of_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters


def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model


def get_pruning_ratio(target_size, model, current_ratio=0.2):
    num_parameters = measure_number_of_parameters(model) - measure_global_sparsity(model)[0]
    if num_parameters * (1 - current_ratio) >= target_size:
        return current_ratio
    else:
        return abs((target_size / num_parameters) - 1)


def get_finetune_iterations(target_size, current_size, ratio):
    numerator = log10(target_size / current_size)
    denominator = log10(1 - ratio)
    return ceil(numerator / denominator)