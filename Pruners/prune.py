# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/prune.py
import os.path

from Utils import config
from Utils.network_utils import eval, pruning_checkpointing, multiplier, get_network
from train import train_imp
from Optimizers.lars import LARS
from Models import Pruners_ResNetModels_CIFAR, Pruners_VGGModels, Pruners_ResNetModels_ImageNet

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb


def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in range(epochs):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity ** ((epoch + 1) / epochs)
            print(f'Sparse: {sparse}')
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)
        remaining_params, total_params = pruner.stats()
        print(f'{remaining_params} remaining params')

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    print(f'{remaining_params} params remaining out of {total_params}')
    if np.abs(remaining_params - total_params * sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))
        quit()


def prune_loop_imp(model, loss, pruner, train_loader, test_loader, device, sparsity, schedule, scope, epochs, path,
                   message, file_name, args, reinitialize=False):
    _, total_params = pruner.stats()
    epoch = 1
    # epoch = epochs
    checkpoint_location = os.path.join(args.checkpoint_dir, 'pruning_checkpoint.pth')
    if os.path.exists(checkpoint_location):
        checkpoint = torch.load(checkpoint_location)
        epoch = checkpoint['epoch']
        rng = checkpoint['rng']
        torch.set_rng_state(rng)
    while epoch < epochs + 1:
        if epoch > 1:
            if not args.reinitialize:
                print(f'Loading {args.model_name.lower()}_{args.expansion_ratio}x_finetune_{epoch - 1}_best.pt')
                model.load_state_dict(torch.load(path + f'{args.model_name.lower()}_{args.expansion_ratio}x_'
                                                        f'finetune_{epoch - 1}_best.pt'))
        model.train()
        configuration = dict(learning_rate=args.lr,
                             dataset=args.dataset,
                             model=args.model_name,
                             train_epochs=args.post_epochs,
                             prune_method=args.pruner)
        wandb.init(project='Pruning Research',
                   config=configuration,
                   entity='sparsetraining')
        wandb.watch(model)
        wandb.run.name = file_name
        wandb.run.save()
        print(f'Pruning ({message}) iteration {epoch} of {epochs}')
        pruner.score(model, loss, train_loader, device)
        params_before_pruning, _ = pruner.stats()
        if schedule == 'exponential':
            sparse = sparsity ** (epoch / epochs)
            print(f'Sparse: {sparse}')
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * (epoch / epochs)
        pruner.mask(sparse, scope)
        remaining_params, total_params = pruner.stats()
        print(f'Before pruning: {params_before_pruning} After pruning: {remaining_params} w/ a pruning ratio of '
              f'{1 - (remaining_params / params_before_pruning)}')
        if args.reinitialize:
            if args.weight_rewind:
                parameter_dict = {}
                batchnorm_dict = {}
                expansion_ratio = args.expansion_ratio
                if 'vgg' in args.model_name.lower():
                    current_cfg = config.defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
                else:
                    if args.dataset.lower() == 'imagenet':
                        current_cfg = config.defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
                    else:
                        current_cfg = config.defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
                multiplier(current_cfg, expansion_ratio)
                temp_model = get_network(args.model_name.lower(), args.dataset, current_cfg)
                if 'vgg' in args.model_name.lower():
                    temp_model.load_state_dict(torch.load(config.PRIVATE_PATH +
                                                          f'/Models/SavedModels/VGG/expansion_ratio_inference/'
                                                          f'{args.model_name.lower()}_{args.expansion_ratio}x_'
                                                          f'for_reinit_{args.weight_rewind_epoch}_epoch.pt'))
                elif 'resnet' in args.model_name.lower():
                    temp_model.load_state_dict(torch.load(config.PRIVATE_PATH +
                                                          f'/Models/SavedModels/ResNet/expansion_ratio_inference/'
                                                          f'{args.model_name.lower()}_{args.expansion_ratio}x_'
                                                          f'{args.weight_rewind_epoch}_epoch.pt'))
                temp_model.to(device)
                for name, param in temp_model.named_parameters():
                    parameter_dict[name] = param
                for name, param in model.named_parameters():
                    param.data = parameter_dict[name]
                # if 'vgg' in args.model_name.lower():
                #     pass
                if 'resnet' in args.model_name.lower() or 'vgg' in args.model_name.lower():
                    for buffer_name, buffer in temp_model.named_buffers():
                        if ('bn' in buffer_name and 'resnet' in args.model_name.lower()) or \
                                ('vgg' in args.model_name.lower() and ('running_mean' in buffer_name or
                                                                       'running_var' in buffer_name or
                                                                       'num_batches_tracked' in buffer_name)):
                            period_positions = [pos for pos, char in enumerate(buffer_name) if char == '.']
                            module_name = buffer_name[:period_positions[-1]]
                            if module_name not in batchnorm_dict:
                                batchnorm_dict[module_name] = {}
                            if 'running_mean' in buffer_name:
                                batchnorm_dict[module_name]['running_mean'] = buffer
                            elif 'running_var' in buffer_name:
                                batchnorm_dict[module_name]['running_var'] = buffer
                            elif 'num_batches_tracked' in buffer_name:
                                batchnorm_dict[module_name]['num_batches_tracked'] = buffer
                    # print(batchnorm_dict)
                    for module_name, module in model.named_modules():
                        if module_name in batchnorm_dict:
                            module.running_mean.data = batchnorm_dict[module_name]['running_mean']
                            module.running_var.data = batchnorm_dict[module_name]['running_var']
                            module.num_batches_tracked.data = batchnorm_dict[module_name]['num_batches_tracked']
                    # print('buffers in temp model')
                    # for buffer_name, buffer in temp_model.named_buffers():
                    #     print((buffer_name, buffer))
                    # print('buffers in new model')
                    # for buffer_name, buffer in model.named_buffers():
                    #     print((buffer_name, buffer))
                print('Weights rewinded')
            else:
                model._initialize_pruned_weights()
                print('Weights reinitialized')
        if args.reinitialize or epoch == epochs:
            if isinstance(model, Pruners_VGGModels.VGG):
                lr = 2.0
            elif isinstance(model, Pruners_ResNetModels_CIFAR.ResNet):
                lr = 2.25
            optimizer = LARS(model.parameters(), lr=lr, max_epoch=args.post_epochs)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=0.1)
            args.post_epochs = 300
        else:
            lr = args.lr * 0.01
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM,
                                  weight_decay=config.WEIGHT_DECAY, nesterov=True)
            scheduler = None
        if (args.shuffle and (args.reinitialize or args.imp_singleshot)) or (args.shuffle and epoch == epochs):
            print('Masks shuffled')
            pruner.shuffle()

        model.eval()
        test_accuracy, _ = eval(model, test_loader, device, None)
        print(f'Test accuracy before training: {test_accuracy}')
        if isinstance(optimizer, LARS):
            print('Optimizing using LARS')
        else:
            print('Optimizing using SGD')

        model.train()
        train_imp(model, train_loader, test_loader, args.post_epochs, optimizer, loss, scheduler,
                  device, path, file_name, epoch, None, args.checkpoint_dir)
        model_save_path = path + file_name + f'_{epoch}_final.pt'
        torch.save(model.state_dict(), model_save_path)
        pruning_checkpointing(epoch, torch.get_rng_state(), args.checkpoint_dir)
        epoch += 1
        print('///////////////////////////////////////////////////////////////////////////////////')
