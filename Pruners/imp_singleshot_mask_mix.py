import os
from collections import OrderedDict

from Utils import config, network_utils, pruning_utils
from Pruners.IMP.finetuning import load_network
from Pruners.prune import prune_loop
from train import train

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def mask_swap(model, mask_model):
    dictionary = OrderedDict()
    illegal_num = -1  # so we don't track BN layers
    for buffer_name, buffer in mask_model.named_buffers():
        layer_number = [int(i) for i in buffer_name.split('.') if i.isdigit()][0]
        if ("running_mean" in buffer_name) or ("running_var" in buffer_name) or \
                ("num_batches_tracked" in buffer_name):
            illegal_num = layer_number
        if 'weight_mask' in buffer_name or 'bias_mask' in buffer_name:
            if ('features' in buffer_name and layer_number != illegal_num) or 'classifier' in buffer_name:
                temp = buffer_name[:buffer_name.rfind('.')]
                if temp in dictionary:
                    dictionary[temp].append((buffer_name, buffer))
                else:
                    dictionary[temp] = [(buffer_name, buffer)]

    for module_name, module in model.named_modules():
        if module_name in dictionary:
            weight_name, weight_buffer, bias_name, bias_buffer = [None] * 4
            for item in dictionary[module_name]:
                if 'weight' in item[0]:
                    weight_name, weight_buffer = item
                else:
                    bias_name, bias_buffer = item
            del module._buffers['weight_mask']
            del module._buffers['bias_mask']
            module.register_buffer('weight_mask', weight_buffer)
            module.register_buffer('bias_mask', bias_buffer)


def run(args):
    found = False
    file_names = {'snip': 'SNIP', 'grasp': 'GraSP', 'synflow': 'SynFlow'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.defaultcfg[11].copy()
    network_utils.multiplier(cfg, args.expansion_ratio)

    temp_path = config.PRIVATE_PATH + f'/Models/SavedModels/{file_names[args.pruner.lower()]}/' \
                                      f'vgg11_{args.expansion_ratio}x_best.pt'
    if os.path.exists(temp_path):
        found = True
        print(f'Old {file_names[args.pruner.lower()]} (w/ expansion ratio {args.expansion_ratio}x) found')
        mask_model = network_utils.get_network('vgg11', 'cifar100', cfg, imp=False)
        mask_model.load_state_dict(torch.load(temp_path))
        mask_model.to(device)
    else:
        mask_model = network_utils.get_network('vgg11', args.dataset, cfg, imp=False).to(device)

    model = load_network('vgg11', args.dataset, cfg, args.expansion_ratio).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    if args.expansion_ratio == 1.0:
        lr = args.lr * 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM,
                                    weight_decay=config.WEIGHT_DECAY, nesterov=True)
        scheduler = None
    elif 1.5 <= args.expansion_ratio < 2.5:
        lr = args.lr * 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM,
                                    weight_decay=config.WEIGHT_DECAY, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    else:
        lr = args.lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM,
                                    weight_decay=config.WEIGHT_DECAY, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 75], gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 0.01, momentum=config.MOMENTUM,
    #                             weight_decay=config.WEIGHT_DECAY, nesterov=True)

    print(f'Loading {args.dataset} dataset')
    input_shape, num_classes = network_utils.dimension(args.dataset)
    if not found:
        prune_loader = network_utils.dataloader(args.dataset, args.prune_batch_size, True,
                                                length=args.prune_dataset_ratio * num_classes)
    train_loader = network_utils.dataloader(args.dataset, args.train_batch_size, True)
    test_loader = network_utils.dataloader(args.dataset, args.test_batch_size, False)

    if not found:
        print(f'Pruning with {args.pruner} for {args.prune_epochs} epochs')
        pruner = pruning_utils.pruner(args.pruner)(pruning_utils.masked_parameters(mask_model))
        _, total_elems = pruner.stats()
        target_sparsity = (config.TARGET_SIZE / total_elems)
        prune_loop(mask_model, loss, pruner, prune_loader, device, target_sparsity, args.compression_schedule, 'global',
                   args.prune_epochs)

    saved_file_name = f'vgg11_{args.expansion_ratio}x_{file_names[args.pruner.lower()]}'
    path_to_best_model = config.PRIVATE_PATH + f'/Models/SavedModels/Finetune_Mask_Mix/{saved_file_name}_best.pt'
    path_to_final_model = config.PRIVATE_PATH + f'/Models/SavedModels/Finetune_Mask_Mix/{saved_file_name}_final.pt'
    path_to_before_train_model = config.PRIVATE_PATH + f'/Models/SavedModels/Finetune_Mask_Mix/' \
                                                       f'{saved_file_name}_before_training.pt '
    if not os.path.isdir(config.PRIVATE_PATH + f'/Models/SavedModels/Finetune_Mask_Mix/'):
        os.mkdir(config.PRIVATE_PATH + f'/Models/SavedModels/Finetune_Mask_Mix/')

    mask_swap(model, mask_model)
    model.to(device)
    writer = SummaryWriter(f'runs/CIFAR100/VGG/Finetune_Mask_Mix/{saved_file_name}')
    torch.save(model.state_dict(), path_to_before_train_model)
    acc_before_training = network_utils.eval(model, test_loader, device, loss)[0]
    print(f"Accuracy before training: {acc_before_training}")
    train(model, train_loader, test_loader, optimizer, scheduler, loss, device, writer, path_to_best_model,
          path_to_final_model, args.post_epochs, args.checkpoint_dir)
