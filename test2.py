import torch
from Pruners.IMP.finetuning import load_network
from Utils.config import defaultcfg_vgg, PRIVATE_PATH
from Utils.network_utils import multiplier, get_network

cfg = defaultcfg_vgg[11].copy()
multiplier(cfg, 1.0)
model_epoch_0 = get_network('vgg11', 'cifar100', cfg)

model_epoch_0.load_state_dict(torch.load(PRIVATE_PATH + '/Models/SavedModels/VGG/expansion_ratio_inference/'
                                                        'vgg11_1.0x_for_reinit_first_epoch.pt'))
model = get_network('vgg11', 'cifar100', cfg)
model.load_state_dict(torch.load(PRIVATE_PATH + '/Models/SavedModels/VGG/expansion_ratio_inference/'
                                                'vgg11_1.0x_for_reinit_best.pt'))
dict = {}
for name, param in model_epoch_0.named_parameters():
    print(name, param)
    dict[name] = param

for name, param in model.named_parameters():
    param.data = dict[name]

for name, param in model.named_parameters():
    print(name, param)

