import os
from collections import OrderedDict

from Utils import config, network_utils, pruning_utils
from Pruners.IMP.finetuning import load_network
from Pruners.prune import prune_loop
from train import train
from Optimizers.lars import LARS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb


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
    file_names = {'snip': 'SNIP', 'grasp': 'GraSP', 'synflow': 'SynFlow'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.defaultcfg_vgg[11].copy()
    network_utils.multiplier(cfg, args.expansion_ratio)

    temp_path = config.PRIVATE_PATH + f'/Models/SavedModels/VGG/{file_names[args.pruner.lower()]}/' \
                                      f'vgg11_{args.expansion_ratio}x_{file_names[args.pruner.lower()]}_best.pt'
    if os.path.exists(temp_path):
        print(f'Old {file_names[args.pruner.lower()]} (w/ expansion ratio {args.expansion_ratio}x) found')
        mask_model = network_utils.get_network('vgg11', 'cifar100', cfg, imp=False)
        mask_model.load_state_dict(torch.load(temp_path))
        mask_model.to(device)
    else:
        print(f'Old mask model with specification not found! Require to train a {file_names[args.pruner.lower()]} model'
              f' with expansion ratio {args.expansion_ratio}x first')
        quit()

    model = load_network('vgg11', args.dataset, cfg, args.expansion_ratio).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = LARS(model.parameters(), lr=args.lr, max_epoch=args.post_epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=0.1)

    print(f'Loading {args.dataset} dataset')
    train_loader = network_utils.dataloader(args.dataset, args.train_batch_size, True)
    test_loader = network_utils.dataloader(args.dataset, args.test_batch_size, False)

    saved_file_name = f'vgg11_{args.expansion_ratio}x_{file_names[args.pruner.lower()]}_MaskMix'
    path_to_best_model = config.PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix/{saved_file_name}_best.pt'
    path_to_final_model = config.PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix/{saved_file_name}_final.pt'
    path_to_before_train_model = config.PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix/' \
                                                       f'{saved_file_name}_before_training.pt '
    if not os.path.isdir(config.PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix/'):
        os.mkdir(config.PRIVATE_PATH + f'/Models/SavedModels/VGG/Finetune_Mask_Mix/')

    mask_swap(model, mask_model)
    model.to(device)
    # writer = SummaryWriter(f'runs/CIFAR100/VGG/Finetune_Mask_Mix/{saved_file_name}')
    configuration = dict(learning_rate=args.lr,
                         dataset=args.dataset,
                         model=args.model_name,
                         train_epochs=args.post_epochs,
                         prune_method=args.pruner)
    wandb.init(project='Pruning Research',
               config=configuration,
               entity='sparsetraining',
               group='singleshot',
               job_type=f'{file_names[args.pruner.lower()]} Maskmixing')
    wandb.watch(model)
    wandb.run.name = saved_file_name
    wandb.run.save()
    torch.save(model.state_dict(), path_to_before_train_model)
    acc_before_training = network_utils.eval(model, test_loader, device, loss)[0]
    print(f"Accuracy before training: {acc_before_training}")
    train(model, train_loader, test_loader, optimizer, scheduler, loss, device, None, path_to_best_model,
          path_to_final_model, args.post_epochs, args.checkpoint_dir)
