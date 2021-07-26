# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/prune.py
import os.path

from Utils import config
from Utils.network_utils import eval, pruning_checkpointing
from train import train_imp
from Optimizers.lars import LARS

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


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
    checkpoint_location = os.path.join(args.checkpoint_dir, 'pruning_checkpoint.pth')
    if os.path.exists(checkpoint_location):
        checkpoint = torch.load(checkpoint_location)
        epoch = checkpoint['epoch']
        rng = checkpoint['rng']
        torch.set_rng_state(rng)
    while epoch < epochs + 1:
        if epoch > 1:
            if not args.reinitialize:
                model.load_state_dict(torch.load(path + f'vgg11_{args.expansion_ratio}x_finetune_{epoch - 1}_best.pt'))
        model.train()
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
            model._initialize_weights()  # TODO make a new reinit weights method
        if not args.reinitialize:
            # lr = args.lr * 0.01
            lr = args.lr
        else:
            lr = args.lr
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM,
        #                       weight_decay=config.WEIGHT_DECAY, nesterov=True)
        optimizer = LARS(model.parameters(), lr=lr, max_epoch=args.post_epochs)
        # if args.reinitialize:
        #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=0.1)
        # else:
        #     scheduler = None
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=0.1)
        summary = SummaryWriter(f'runs/CIFAR100/VGG/{message}/{file_name}_{epoch}')
        model.eval()
        test_accuracy, _ = eval(model, test_loader, device, None)
        print(f'Test accuracy before training: {test_accuracy}')

        model.train()
        train_imp(model, train_loader, test_loader, args.post_epochs, optimizer, loss, scheduler,
                  device, path, file_name, epoch, summary, args.checkpoint_dir)
        model_save_path = path + file_name + f'_{epoch}_final.pt'
        torch.save(model.state_dict(), model_save_path)
        pruning_checkpointing(epoch, torch.get_rng_state(), args.checkpoint_dir)
        epoch += 1
        print('///////////////////////////////////////////////////////////////////////////////////')
