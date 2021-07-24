import torch
from Pruners.IMP.finetuning import load_network
from Utils.config import defaultcfg
from Utils.network_utils import multiplier
from Layers import layers
from Pruners.imp_singleshot_mask_mix import mask_swap

cfg = defaultcfg[11].copy()
multiplier(cfg, 4.0)
model = load_network('vgg11', 'cifar100', cfg, 4.0)
# for buffer_name, buffer in model.named_buffers():
#     print(buffer_name)
#
# for module_name, module in model.named_modules():
#     print(module_name)

# dictionary = mask_swap(None, model)

# print(dictionary.keys())
# print(dictionary)
for module_name, module in model.named_modules():
    # for buffer_name in dictionary.keys():
        # if module_name != 'features' and module_name != 'classifier':
    # if module_name in dictionary:
    #     print(module_name)
        # break
    # print(type(module))
    if isinstance(module, layers.Conv2d):
        print(f'Weight shape: {module.weight.size()}')
        print(f'Bias shape: {module.bias.size()}')
