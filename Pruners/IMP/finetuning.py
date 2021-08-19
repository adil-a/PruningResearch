import os

from torch.nn.utils import prune
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb

from Utils.config import PRIVATE_PATH, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE, SEED, VGG_TARGET_SIZE, defaultcfg_vgg, \
    defaultcfg_resnet_cifar, defaultcfg_resnet_imagenet, RESNET_CIFAR_TARGET_SIZE
from Utils import pruning_utils
from Utils.network_utils import multiplier, get_network, dataloader, eval
from Models import Pruners_VGGModels, Pruners_ResNetModels_CIFAR, Pruners_ResNetModels_ImageNet
from Pruners.prune import prune_loop_imp


def load_network(net_type, dataset, config, expansion_rate, args):
    net = get_network(net_type, dataset, config)
    if 'vgg' in args.model_name.lower():
        best_model_path = PRIVATE_PATH + f'/Models/SavedModels/VGG/expansion_ratio_inference/' \
                                         f'{args.model_name.lower()}_{expansion_rate}x_best.pt'
    elif 'resnet' in args.model_name.lower():
        best_model_path = PRIVATE_PATH + f'/Models/SavedModels/ResNet/expansion_ratio_inference/' \
                                         f'{args.model_name.lower()}_{expansion_rate}x_best.pt'
    net.load_state_dict(torch.load(best_model_path))
    return net


def main(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.login()

    expansion_ratio = args.expansion_ratio
    learning_rate = args.lr
    if 'vgg' in args.model_name.lower():
        current_cfg = defaultcfg_vgg[int(args.model_name.lower().replace('vgg', ''))].copy()
    else:
        if args.dataset.lower() == 'imagenet':
            current_cfg = defaultcfg_resnet_imagenet[int(args.model_name.lower().replace('resnet', ''))].copy()
        else:
            current_cfg = defaultcfg_resnet_cifar[int(args.model_name.lower().replace('resnet', ''))].copy()
    multiplier(current_cfg, expansion_ratio)
    print(f'Current {args.model_name.upper()} config being used: {current_cfg} (ratio {expansion_ratio}x) '
          f'(Batchsize: {BATCH_SIZE}, LR: {learning_rate})')
    # if args.lr_rewind:
    #     message = 'Finetuning(LR Rewinding)'
    #     saved_file_name = f'vgg11_{expansion_ratio}x_lr_rewind'
    #     if not os.path.isdir(
    #             PRIVATE_PATH + '/Models/SavedModels/LR_Rewind'):
    #         os.mkdir(PRIVATE_PATH + '/Models/SavedModels/LR_Rewind')
    #     path = PRIVATE_PATH + '/Models/SavedModels/LR_Rewind/'
    if args.reinitialize:
        if args.shuffle:
            message = 'Reinitializing w/ mask shuffle'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_reinitialize_Shuffled'
        elif args.weight_rewind:
            message = 'Reinitializing w/ weight rewinding to 0th epoch'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_reinitialize_weight_rewind'
        else:
            message = 'Reinitializing'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_reinitialize'
        if 'vgg' in args.model_name.lower():
            if not os.path.isdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Reinitialize'):
                os.mkdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Reinitialize')
            path = PRIVATE_PATH + '/Models/SavedModels/VGG/Reinitialize/'
        elif 'resnet' in args.model_name.lower():
            if not os.path.isdir(PRIVATE_PATH + '/Models/SavedModels/ResNet/Reinitialize'):
                os.mkdir(PRIVATE_PATH + '/Models/SavedModels/ResNet/Reinitialize')
            path = PRIVATE_PATH + '/Models/SavedModels/ResNet/Reinitialize/'
    elif args.imp_singleshot:
        if args.shuffle:
            message = 'Finetuning (Singleshot IMP) w/ mask shuffle'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_finetune_singleshot_Shuffled'
        else:
            message = 'Finetuning (Singleshot IMP)'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_finetune_singleshot'
        if 'vgg' in args.model_name.lower():
            if not os.path.isdir(
                    PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune_Singleshot'):
                os.mkdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune_Singleshot')
            path = PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune_Singleshot/'
        elif 'resnet' in args.model_name.lower():
            if not os.path.isdir(
                    PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune_Singleshot'):
                os.mkdir(PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune_Singleshot')
            path = PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune_Singleshot/'
    else:
        if args.shuffle:
            message = 'Finetuning w/ mask shuffle on last iteration of pruning'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_finetune_Shuffled'
        else:
            message = 'Finetuning'
            saved_file_name = f'{args.model_name.lower()}_{expansion_ratio}x_finetune'
        if 'vgg' in args.model_name.lower():
            if not os.path.isdir(
                    PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune'):
                os.mkdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune')
            path = PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune/'
        elif 'resnet' in args.model_name.lower():
            if not os.path.isdir(
                    PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune'):
                os.mkdir(PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune')
            path = PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune/'

    trainloader = dataloader('cifar100', BATCH_SIZE, True)
    testloader = dataloader('cifar100', BATCH_SIZE, False)

    net = load_network(args.model_name.lower(), args.dataset.lower(), current_cfg, expansion_ratio, args)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    pruner = pruning_utils.pruner(args.pruner)(pruning_utils.masked_parameters(net))
    if isinstance(net, Pruners_VGGModels.VGG):
        pruning_iters = pruning_utils.get_finetune_iterations(VGG_TARGET_SIZE,
                                                              pruner.stats()[1], 0.2)
    elif isinstance(net, Pruners_ResNetModels_CIFAR.ResNet):
        pruning_iters = pruning_utils.get_finetune_iterations(RESNET_CIFAR_TARGET_SIZE,
                                                              pruner.stats()[1], 0.2)
    _, total_elems = pruner.stats()
    if isinstance(net, Pruners_VGGModels.VGG):
        target_sparsity = (VGG_TARGET_SIZE / total_elems)
    elif isinstance(net, Pruners_ResNetModels_CIFAR.ResNet):
        target_sparsity = (RESNET_CIFAR_TARGET_SIZE / total_elems)
    # if args.lr_rewind:
    #     pruning_finetuning(net, trainloader, testloader, device, pruning_iters, 200, 0.2, TARGET_SIZE,
    #                        criterion, path, saved_file_name, message, learning_rate, args.lr_rewind,
    #                        args.reinitialize, expansion_ratio)
    if args.reinitialize:
        if isinstance(net, Pruners_VGGModels.VGG):
            temp = PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune/' + f'{args.model_name.lower()}_' \
                                                                        f'{expansion_ratio}x_finetune_' \
                                                                        f'{pruning_iters - 1}_best.pt'
            assert os.path.isfile(temp), f"required file {temp} does not exist"
        elif isinstance(net, Pruners_ResNetModels_CIFAR.ResNet):
            temp = PRIVATE_PATH + '/Models/SavedModels/ResNet/Finetune/' + f'{args.model_name.lower()}_' \
                                                                           f'{expansion_ratio}x_finetune_' \
                                                                           f'{pruning_iters - 1}_best.pt'
            assert os.path.isfile(temp), f"required file {temp} does not exist"

        net.load_state_dict(torch.load(temp))
        prune_loop_imp(net, criterion, pruner, trainloader, testloader, device, target_sparsity,
                       args.compression_schedule, 'global', 1, path, message, saved_file_name, args, True)
    elif args.imp_singleshot:
        prune_loop_imp(net, criterion, pruner, trainloader, testloader, device, target_sparsity,
                       args.compression_schedule, 'global', 1, path, message, saved_file_name, args)
    else:
        prune_loop_imp(net, criterion, pruner, trainloader, testloader, device, target_sparsity,
                       args.compression_schedule, 'global', pruning_iters, path, message, saved_file_name,
                       args)
