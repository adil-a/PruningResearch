# code from https://leimao.github.io/blog/PyTorch-Pruning/
import os
import argparse

from torch.nn.utils import prune
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import PRIVATE_PATH, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE
from OverparameterizationVerification import val, get_lr
from Utils import pruning_utils
from Utils.network_utils import multiplier, get_network
from Models.VGGModels import weights_init

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

TARGET_SIZE = 28518244  # gotten from number of trainable parameters in VGG11 w/ expansion ratio of 1.0x


def load_network(net_type, dataset, config, expansion_rate):
    net = get_network(net_type, dataset, config)
    best_model_path = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/vgg11_{expansion_rate}x_best.pt'
    net.load_state_dict(torch.load(best_model_path))
    return net


def train(network, train_data, test_data, epochs, optimizer, criterion, scheduler,
          device, path, file_name, pruning_iteration, writer):
    network.train()
    step = 0
    curr_best_accuracy = 0
    curr_best_epoch = 0
    for epoch in range(1, epochs + 1):
        current_loss = 0
        print(f'Epoch {epoch} of {epochs}')
        for batch_idx, (data, targets) in enumerate(train_data):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = network(data)
            loss = criterion(scores, targets)
            current_loss += loss
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        network.eval()
        curr_test_accuracy, curr_test_loss = val(network, test_data, device, criterion)
        curr_training_accuracy, _ = val(network, train_data, device, criterion)
        if curr_test_accuracy > curr_best_accuracy:
            curr_best_accuracy = curr_test_accuracy
            curr_best_epoch = epoch
            model_save_path = path + file_name + f'_{pruning_iteration + 1}_best.pt'
            torch.save(network.state_dict(), model_save_path)
        network.train()
        print(f'Current accuracy: {curr_test_accuracy}')
        print(f'Loss: {current_loss.item() / len(train_data)}')
        print(f'LR: {optimizer.param_groups[0]["lr"]}')
        writer.add_scalar('Training Loss', current_loss.item() / len(train_data), global_step=step)
        writer.add_scalar('Training Accuracy', curr_training_accuracy, global_step=step)
        writer.add_scalar('Test Loss', curr_test_loss, global_step=step)
        writer.add_scalar('Test Accuracy', curr_test_accuracy, global_step=step)
        step += 1
        writer.flush()
        print('--------------------------------------------------')
    print(f'Best accuracy was {curr_best_accuracy} at epoch {curr_best_epoch}')


def pruning_finetuning(model, train_loader, test_loader, device, pruning_iterations, finetune_epochs,
                       current_ratio, target_size, criterion, path, file_name, message, lr, rewind, reinit):
    initial_number_of_parameters = pruning_utils.measure_number_of_parameters(model)
    for i in range(pruning_iterations):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY, nesterov=True)
        if rewind or reinit:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)
        else:
            scheduler = None
        summary = SummaryWriter(f'runs/CIFAR100/VGG/{message}/{file_name}_{i + 1}')
        print(f'Pruning / {message} iteration {i + 1} of {pruning_iterations}')
        print('Pruning')
        if i == 0:
            print(f'Number of parameters before pruning {pruning_utils.measure_number_of_parameters(model)}')
        else:
            print(f'Number of parameters before pruning {initial_number_of_parameters - global_sparsity[0]}')
        ratio = pruning_utils.get_pruning_ratio(target_size, model, current_ratio)
        print(f'Pruning by ratio of {ratio}')

        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
                parameters_to_prune.append((module, "bias"))

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=ratio)
        global_sparsity = pruning_utils.measure_global_sparsity(model)
        print(f'Number of parameters after pruning {initial_number_of_parameters - global_sparsity[0]} out of '
              f'{initial_number_of_parameters} (Sparsity {global_sparsity[2]})')
        model.eval()
        test_accuracy, _ = val(model, test_loader, device, None)
        print(f'Test accuracy before training: {test_accuracy}')

        print(f'{message}')
        if reinit:
            model.apply(weights_init)
        model.train()
        train(model, train_loader, test_loader, finetune_epochs, optimizer, criterion, scheduler,
              device, path, file_name, i, summary)
        model.eval()
        test_accuracy, _ = val(model, test_loader, device, None)
        print(f'Test accuracy after training: {test_accuracy}')
        model.train()

        print('///////////////////////////////////////////////////////////////////////////////////')
        model_save_path = path + file_name + f'_{i + 1}_final.pt'
        torch.save(model.state_dict(), model_save_path)
    return model


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('-expansion_ratio', type=float, required=True)
    parser.add_argument('-finetune_epochs', type=int, required=True)
    parser.add_argument('-lr_rewind', type=bool, default=False, required=False)
    parser.add_argument('-reinitialize', type=bool, default=False, required=False)
    args = parser.parse_args()

    expansion_ratio = args.expansion_ratio
    learning_rate = get_lr(expansion_ratio)
    if args.lr_rewind or args.reinitialize:
        learning_rate = learning_rate * 1
    else:
        learning_rate = learning_rate * 0.01
    current_cfg = defaultcfg[11].copy()
    multiplier(current_cfg, expansion_ratio)
    print(f'Current VGG11 config being used: {current_cfg} (ratio {expansion_ratio}x) (Batchsize: {BATCH_SIZE})')
    if args.lr_rewind:
        message = 'Finetuning(LR Rewinding)'
        saved_file_name = f'vgg11_{expansion_ratio}x_lr_rewind'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/LR_Rewind'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/LR_Rewind')
        path = PRIVATE_PATH + '/Models/SavedModels/LR_Rewind/'
    elif args.reinitialize:
        message = 'Reinitializing'
        saved_file_name = f'vgg11_{expansion_ratio}x_reinitialize'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/Reinitialize'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/Reinitialize')
        path = PRIVATE_PATH + '/Models/SavedModels/Reinitialize/'
    else:
        message = 'Finetuning'
        saved_file_name = f'vgg11_{expansion_ratio}x_finetune'
        if not os.path.isdir(
                PRIVATE_PATH + '/Models/SavedModels/Finetune'):
            os.mkdir(PRIVATE_PATH + '/Models/SavedModels/Finetune')
        path = PRIVATE_PATH + '/Models/SavedModels/Finetune/'

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomRotation(15)]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
    )

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    net = load_network('vgg11', 'cifar100', current_cfg, expansion_ratio)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    pruning_iters = pruning_utils.get_finetune_iterations(TARGET_SIZE,
                                                          pruning_utils.measure_number_of_parameters(net), 0.2)

    if args.lr_rewind or args.reinitialize:
        pruning_finetuning(net, trainloader, testloader, device, pruning_iters, 200, 0.2, TARGET_SIZE,
                           criterion, path, saved_file_name, message, learning_rate, args.lr_rewind, args.reinitialize)
    else:
        pruning_finetuning(net, trainloader, testloader, device, pruning_iters, args.finetune_epochs, 0.2, TARGET_SIZE,
                           criterion, path, saved_file_name, message, learning_rate, args.lr_rewind, args.reinitialize)


if __name__ == '__main__':
    main()
