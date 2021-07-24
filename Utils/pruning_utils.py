# code taken from https://leimao.github.io/blog/PyTorch-Pruning/
import torch
from torch.nn.utils import prune
from math import ceil, log10

from Layers import layers
from Pruners import pruners


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
        if isinstance(module, layers.Conv2d) or isinstance(module, layers.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(module)
            total_num_zeros += module_num_zeros
            total_num_elements += module_num_elements
    if total_num_elements != 0:
        sparsity = total_num_zeros / total_num_elements
    else:
        sparsity = 0
    return total_num_zeros, total_num_elements, sparsity


def measure_number_of_parameters(model):
    # num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return num_parameters
    total = 0
    for module_name, module in model.named_modules():
        if isinstance(module, (layers.Conv2d, layers.Linear)):
            total += sum(p.numel() for p in module.parameters())
    return total

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


# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/Utils/generator.py


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (layers.Identity1d, layers.Identity2d))


def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (layers.Linear, layers.Conv2d))
    if batchnorm:
        isprunable |= isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (layers.Identity1d, layers.Identity2d))
    return isprunable


def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param


def masked_parameters(model, bias=True, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param


def pruner(method):
    method = method.lower()
    prune_methods = {
        'rand': pruners.Rand,
        'mag': pruners.Mag,
        'snip': pruners.SNIP,
        'grasp': pruners.GraSP,
        'synflow': pruners.SynFlow,
    }
    return prune_methods[method]


def stats(model):
    r"""Returns remaining and total number of prunable parameters.
    """
    remaining_params, total_params = 0, 0
    for mask, _ in model.masked_parameters:
        remaining_params += mask.detach().cpu().numpy().sum()
        total_params += mask.numel()
    return remaining_params, total_params
