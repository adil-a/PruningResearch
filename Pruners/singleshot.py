# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/Experiments/singleshot.py
import os.path

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from Utils import network_utils, pruning_utils, config
from Pruners.prune import prune_loop
from train import train
from Optimizers.lars import LARS
from Models import Pruners_VGGModels, Pruners_ResNetModels_CIFAR
import wandb


def run(args):
    file_names = {'snip': 'SNIP', 'grasp': 'GraSP', 'synflow': 'SynFlow'}
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Loading {args.dataset} dataset')
    input_shape, num_classes = network_utils.dimension(args.dataset)
    prune_loader = network_utils.dataloader(args.dataset, args.prune_batch_size, True,
                                            length=args.prune_dataset_ratio * num_classes)
    train_loader = network_utils.dataloader(args.dataset, args.train_batch_size, True)
    test_loader = network_utils.dataloader(args.dataset, args.test_batch_size, False)

    if 'vgg' in args.model_name.lower():
        cfg = config.defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
    else:
        if args.dataset.lower() == 'imagenet':
            cfg = config.defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
        else:
            cfg = config.defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
    network_utils.multiplier(cfg, args.expansion_ratio)
    model = network_utils.get_network(args.model_name, args.dataset, cfg).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = LARS(model.parameters(), lr=args.lr, max_epoch=args.post_epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=0.1)

    if args.shuffle:
        saved_file_name = f'{args.model_name.lower()}_{args.expansion_ratio}x_{file_names[args.pruner.lower()]}_' \
                          f'Shuffled'
    else:
        saved_file_name = f'{args.model_name.lower()}_{args.expansion_ratio}x_{file_names[args.pruner.lower()]}'
    configuration = dict(learning_rate=args.lr,
                         dataset=args.dataset,
                         model=args.model_name,
                         train_epochs=args.post_epochs,
                         prune_method=args.pruner)
    wandb.init(project='Pruning Research',
               config=configuration,
               entity='sparsetraining')
    wandb.watch(model)
    wandb.run.name = saved_file_name
    wandb.run.save()

    print(f'Pruning with {args.pruner} for {args.prune_epochs} epochs')
    pruner = pruning_utils.pruner(args.pruner)(pruning_utils.masked_parameters(model))
    _, total_elems = pruner.stats()
    print(f'Total params: {total_elems}')
    if isinstance(model, Pruners_VGGModels.VGG):
        target_sparsity = (config.VGG_TARGET_SIZE / total_elems)
    elif isinstance(model, Pruners_ResNetModels_CIFAR.ResNet):
        target_sparsity = (config.RESNET_CIFAR_TARGET_SIZE / total_elems)
    prune_loop(model, loss, pruner, prune_loader, device, target_sparsity, args.compression_schedule, 'global',
               args.prune_epochs)
    if args.shuffle:
        print('Masks shuffled')
        pruner.shuffle()

    if 'vgg' in args.model_name.lower():
        path_to_best_model = config.PRIVATE_PATH + f'/Models/SavedModels/VGG/{file_names[args.pruner.lower()]}/' \
                                                   f'{saved_file_name}_best.pt'
        path_to_final_model = config.PRIVATE_PATH + f'/Models/SavedModels/VGG/{file_names[args.pruner.lower()]}/' \
                                                    f'{saved_file_name}_final.pt'
        if not os.path.isdir(config.PRIVATE_PATH + f'/Models/SavedModels/VGG/{file_names[args.pruner.lower()]}/'):
            os.mkdir(config.PRIVATE_PATH + f'/Models/SavedModels/VGG/{file_names[args.pruner.lower()]}/')
    elif 'resnet' in args.model_name.lower():
        path_to_best_model = config.PRIVATE_PATH + f'/Models/SavedModels/ResNet/{file_names[args.pruner.lower()]}/' \
                                                   f'{saved_file_name}_best.pt'
        path_to_final_model = config.PRIVATE_PATH + f'/Models/SavedModels/ResNet/{file_names[args.pruner.lower()]}/' \
                                                    f'{saved_file_name}_final.pt'
        if not os.path.isdir(config.PRIVATE_PATH + f'/Models/SavedModels/ResNet/{file_names[args.pruner.lower()]}/'):
            os.mkdir(config.PRIVATE_PATH + f'/Models/SavedModels/ResNet/{file_names[args.pruner.lower()]}/')

    # print('Weights reinitialized')
    # model._initialize_pruned_weights()  # TODO remove after finishing training new initialization models
    train(model, train_loader, test_loader, optimizer, scheduler, loss, device, None, path_to_best_model,
          path_to_final_model, args.post_epochs, args.checkpoint_dir)
