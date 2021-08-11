import os

from torch.nn.utils import prune
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb

from Utils.config import PRIVATE_PATH, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE, SEED, TARGET_SIZE, defaultcfg_vgg
from Utils import pruning_utils
from Utils.network_utils import multiplier, get_network, dataloader, eval
from Models.IMP_VGGModels import weights_init
from Pruners.prune import prune_loop_imp


def load_network(net_type, dataset, config, expansion_rate):
    net = get_network(net_type, dataset, config, imp=False)
    best_model_path = PRIVATE_PATH + f'/Models/SavedModels/VGG/expansion_ratio_inference/vgg11_{expansion_rate}x_best.pt'
    net.load_state_dict(torch.load(best_model_path))
    return net


def main(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.login()

    expansion_ratio = args.expansion_ratio
    learning_rate = args.lr
    current_cfg = defaultcfg_vgg[11].copy()
    multiplier(current_cfg, expansion_ratio)
    print(f'Current VGG11 config being used: {current_cfg} (ratio {expansion_ratio}x) (Batchsize: {BATCH_SIZE}, '
          f'LR: {learning_rate})')
    # if args.lr_rewind:
    #     message = 'Finetuning(LR Rewinding)'
    #     saved_file_name = f'vgg11_{expansion_ratio}x_lr_rewind'
    #     if not os.path.isdir(
    #             PRIVATE_PATH + '/Models/SavedModels/LR_Rewind'):
    #         os.mkdir(PRIVATE_PATH + '/Models/SavedModels/LR_Rewind')
    #     path = PRIVATE_PATH + '/Models/SavedModels/LR_Rewind/'
    if args.reinitialize:
        message = 'Reinitializing'
        saved_file_name = f'vgg11_{expansion_ratio}x_reinitialize'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/VGG/Reinitialize'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Reinitialize')
        path = PRIVATE_PATH + '/Models/SavedModels/VGG/Reinitialize/'
    elif args.imp_singleshot:
        message = 'Finetuning (Singleshot IMP)'
        saved_file_name = f'vgg11_{expansion_ratio}x_finetune_singleshot'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune_Singleshot'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune_Singleshot')
        path = PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune_Singleshot/'
    else:
        message = 'Finetuning'
        saved_file_name = f'vgg11_{expansion_ratio}x_finetune'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune')
        path = PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune/'

    trainloader = dataloader('cifar100', BATCH_SIZE, True)
    testloader = dataloader('cifar100', BATCH_SIZE, False)

    net = load_network('vgg11', 'cifar100', current_cfg, expansion_ratio)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    pruner = pruning_utils.pruner(args.pruner)(pruning_utils.masked_parameters(net))
    pruning_iters = pruning_utils.get_finetune_iterations(TARGET_SIZE,
                                                         pruner.stats()[1], 0.2)
    _, total_elems = pruner.stats()
    target_sparsity = (TARGET_SIZE / total_elems)
    # if args.lr_rewind:
    #     pruning_finetuning(net, trainloader, testloader, device, pruning_iters, 200, 0.2, TARGET_SIZE,
    #                        criterion, path, saved_file_name, message, learning_rate, args.lr_rewind,
    #                        args.reinitialize, expansion_ratio)
    if args.reinitialize:
        temp = PRIVATE_PATH + '/Models/SavedModels/VGG/Finetune/' + f'vgg11_{expansion_ratio}x_finetune' \
                                                                f'_{pruning_iters - 1}_best.pt'
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
