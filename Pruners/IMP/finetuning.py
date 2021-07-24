import os

from torch.nn.utils import prune
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Utils.config import PRIVATE_PATH, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE, SEED, TARGET_SIZE
from Utils import pruning_utils
from Utils.network_utils import multiplier, get_network, dataloader, eval
from Models.IMP_VGGModels import weights_init
from Pruners.prune import prune_loop_imp

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def load_network(net_type, dataset, config, expansion_rate):
    net = get_network(net_type, dataset, config, imp=False)
    best_model_path = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/vgg11_{expansion_rate}x_best.pt'
    net.load_state_dict(torch.load(best_model_path))
    return net


# def pruning_finetuning(model, train_loader, test_loader, device, pruning_iterations, finetune_epochs,
#                        current_ratio, target_size, criterion, path, file_name, message, lr, rewind, reinit,
#                        expansion_ratio):
#     initial_number_of_parameters = pruning_utils.measure_number_of_parameters(model)
#
#     if not rewind and not reinit:
#         lr = lr * 0.01
#
#     for i in range(pruning_iterations):
#         if i != 0:
#             if rewind:
#                 model.load_state_dict(torch.load(path + f'vgg11_{expansion_ratio}x_lr_rewind_{i}_best.pt'))
#             else:
#                 model.load_state_dict(torch.load(path + f'vgg11_{expansion_ratio}x_finetune_{i}_best.pt'))
#             model.train()
#
#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM,
#                               weight_decay=WEIGHT_DECAY, nesterov=True)
#         if rewind or reinit:
#             scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)
#         else:
#             scheduler = None
#         summary = SummaryWriter(f'runs/CIFAR100/VGG/{message}/{file_name}_{i + 1}')
#         print(f'Pruning / {message} iteration {i + 1} of {pruning_iterations}')
#         print('Pruning')
#         if i == 0:
#             print(f'Number of parameters before pruning {pruning_utils.measure_number_of_parameters(model)}')
#         else:
#             print(f'Number of parameters before pruning {initial_number_of_parameters - global_sparsity[0]}')
#         ratio = pruning_utils.get_pruning_ratio(target_size, model, current_ratio)
#         print(f'Pruning by ratio of {ratio}')
#
#         parameters_to_prune = []
#         for module_name, module in model.named_modules():
#             if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#                 parameters_to_prune.append((module, "weight"))
#                 parameters_to_prune.append((module, "bias"))
#
#         prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=ratio)
#         global_sparsity = pruning_utils.measure_global_sparsity(model)
#         print(f'Number of parameters after pruning {initial_number_of_parameters - global_sparsity[0]} out of '
#               f'{initial_number_of_parameters} (Sparsity {global_sparsity[2]})')
#         model.eval()
#         test_accuracy, _ = eval(model, test_loader, device, None)
#         print(f'Test accuracy before training: {test_accuracy}')
#
#         print(f'{message}')
#         if reinit:
#             model.apply(weights_init)
#         model.train()
#         train(model, train_loader, test_loader, finetune_epochs, optimizer, criterion, scheduler,
#               device, path, file_name, i, summary)
#         model.eval()
#         test_accuracy, _ = eval(model, test_loader, device, None)
#         print(f'Test accuracy after training: {test_accuracy}')
#         model.train()
#
#         print('///////////////////////////////////////////////////////////////////////////////////')
#         model_save_path = path + file_name + f'_{i + 1}_final.pt'
#         torch.save(model.state_dict(), model_save_path)
#     return model


def main(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expansion_ratio = args.expansion_ratio
    learning_rate = args.lr
    current_cfg = defaultcfg[11].copy()
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
                PRIVATE_PATH + '/Models/SavedModels/Reinitialize'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/Reinitialize')
        path = PRIVATE_PATH + '/Models/SavedModels/Reinitialize/'
    elif args.imp_singleshot:
        message = 'Finetuning (Singleshot IMP)'
        saved_file_name = f'vgg11_{expansion_ratio}x_finetune_singleshot'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/Finetune_Singleshot'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/Finetune_Singleshot')
        path = PRIVATE_PATH + '/Models/SavedModels/Finetune_Singleshot/'
    else:
        message = 'Finetuning'
        saved_file_name = f'vgg11_{expansion_ratio}x_finetune'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/Finetune'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/Finetune')
        path = PRIVATE_PATH + '/Models/SavedModels/Finetune/'

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
        temp = PRIVATE_PATH + '/Models/SavedModels/Finetune/' + f'vgg11_{expansion_ratio}x_finetune' \
                                                                f'_{pruning_iters - 1}_best.pt'
        assert os.path.isfile(temp), f"required file {temp} does not exist"

        # for module_name, module in net.named_modules():
        #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        #         prune.identity(module, "weight")
        #         prune.identity(module, "bias")

        net.load_state_dict(torch.load(temp))
        # pruning_finetuning(net, trainloader, testloader, device, 1, 200, 0.2, TARGET_SIZE, criterion, path,
        #                    saved_file_name, message, learning_rate, args.lr_rewind, args.reinitialize, expansion_ratio)
        prune_loop_imp(net, criterion, pruner, trainloader, testloader, device, target_sparsity,
                       args.compression_schedule, 'global', 1, path, message, saved_file_name, args, True)
    elif args.imp_singleshot:
        prune_loop_imp(net, criterion, pruner, trainloader, testloader, device, target_sparsity,
                       args.compression_schedule, 'global', 1, path, message, saved_file_name, args)
    else:
        # pruning_finetuning(net, trainloader, testloader, device, pruning_iters, args.finetune_epochs, 0.2, TARGET_SIZE,
        #                    criterion, path, saved_file_name, message, learning_rate, args.lr_rewind, args.reinitialize,
        #                    expansion_ratio)
        prune_loop_imp(net, criterion, pruner, trainloader, testloader, device, target_sparsity,
                       args.compression_schedule, 'global', pruning_iters, path, message, saved_file_name,
                       args)
