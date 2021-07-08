import argparse
import os

from config import PRIVATE_PATH, BATCH_SIZE, EPOCHS, LR, MOMENTUM, WEIGHT_DECAY
from Utils.network_utils import get_network, get_train_valid_loader, get_test_loader, multiplier, get_lr_array

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch

FIND_BASELINE = False  # used in validation to find a baseline that fully fits our training data
defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def del_file(path):
    if os.path.isfile(path):
        os.remove(path)


def train(network, train_data, val_data, optimizer, scheduler, criterion, device, writer, path, path_final_epoch,
          epochs):
    curr_best_accuracy = 0
    best_accuracy_epoch = 0
    step = 0
    network.train()
    for epoch in range(1, epochs + 1):
        current_loss = 0
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} of {epochs}")
        for batch_idx, (data, targets) in enumerate(train_data):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = network(data)
            loss = criterion(scores, targets)
            current_loss += loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        network.eval()
        curr_test_accuracy, curr_test_loss = val(network, val_data, device, criterion, validate=False)
        curr_accuracy, _ = val(network, train_data, device, criterion)
        network.train()
        if curr_test_accuracy > curr_best_accuracy:
            curr_best_accuracy = curr_test_accuracy
            torch.save(network.state_dict(), path)
            best_accuracy_epoch = epoch
        print(f'Current accuracy: {curr_test_accuracy}')
        print(f'Loss: {current_loss.item() / len(train_data)}')
        print(f'LR: {curr_lr}')
        writer.add_scalar('Training Loss', current_loss.item() / len(train_data), global_step=step)
        writer.add_scalar('Training Accuracy', curr_accuracy, global_step=step)
        writer.add_scalar('Validation Loss', curr_test_loss, global_step=step)
        writer.add_scalar('Validation Accuracy', curr_test_accuracy, global_step=step)
        step += 1
        writer.flush()
        print('--------------------------------------------------')
    print(f'Best accuracy was {curr_best_accuracy} at epoch {best_accuracy_epoch}')
    torch.save(network.state_dict(), path_final_epoch)
    writer.close()


def val(network, val_data, device, criterion, validate=False, validate_amount=0.1):
    correct = 0
    if validate:
        total = len(val_data.dataset) * validate_amount
    else:
        total = len(val_data.dataset)
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_data):
            data, targets = data.to(device), targets.to(device)
            out = network(data)
            if criterion is not None:
                loss = criterion(out, targets)
                test_loss += loss
            _, preds = out.max(1)
            correct += preds.eq(targets).sum()
    return correct / total, test_loss / len(val_data)


def get_lr(ratio: float):
    new_lr = LR / ratio
    if new_lr > 0.1:
        return 0.1
    else:
        return new_lr


def main():
    print(f'Number of GPUs being used: {torch.cuda.device_count()}')
    parser = argparse.ArgumentParser()
    parser.add_argument('-ratio', type=float, default=1)
    parser.add_argument('-epochs', type=int, default=EPOCHS, required=False)
    parser.add_argument('-lr', type=float, required=True)
    args = parser.parse_args()
    torch.manual_seed(1)

    current_ratio = args.ratio
    current_lr = args.lr
    # lr_array = get_lr_array(0.1 / current_ratio, 0.1)
    # print(f'List of LRs: {lr_array}')
    current_cfg = defaultcfg[11]
    multiplier(current_cfg, current_ratio)

    # old_dr_best = ''
    # old_dr_final = ''
    # top_acc = 0
    # best_lr = 0
    # best_lr_acc = 0

    trainloader, _ = get_train_valid_loader(BATCH_SIZE, False)
    testloader = get_test_loader(BATCH_SIZE)
    # for lr in lr_array:
    print(f'Current VGG11 config being used: {current_cfg} (ratio {current_ratio}x) (Batchsize: {BATCH_SIZE}, '
          f'LR: {current_lr})')
    saved_file_name = f'vgg11_{current_ratio}x'
    PATH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/{saved_file_name}_best.pt'
    PATH_FINAL_EPOCH = PRIVATE_PATH + f'/Models/SavedModels/expansion_ratio_inference/' \
                                      f'{saved_file_name}_final_epoch.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f'runs/CIFAR100/VGG/{saved_file_name}')

    net = get_network('vgg11', 'cifar100', current_cfg)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if FIND_BASELINE:
        train(net, trainloader, trainloader, optimizer, scheduler, criterion, device, writer, PATH,
                    PATH_FINAL_EPOCH, args.epochs)
    else:
        train(net, trainloader, testloader, optimizer, scheduler, criterion, device, writer, PATH,
                    PATH_FINAL_EPOCH, args.epochs)
    # if acc > top_acc:
    #     top_acc = acc
    #     best_lr = lr
    #     best_lr_acc = val(net, testloader, device, None)[0]  # this is evaluated on the final model and not the best
    #     # performing model
    #     del_file(old_dr_best)
    #     del_file(old_dr_final)
    #     old_dr_best = PATH
    #     old_dr_final = PATH_FINAL_EPOCH
    # else:
    #     del_file(PATH)
    #     del_file(PATH_FINAL_EPOCH)
    # print(f'Best LR is {best_lr} with test accuracy {best_lr_acc}')


if __name__ == '__main__':
    main()
