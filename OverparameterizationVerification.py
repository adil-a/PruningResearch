# this is still a very sandbox version of what the final file should look like
# hyperparameters based on https://github.com/weiaicunzai/pytorch-cifar100
import os
import argparse

from Utils.Network_Retrieval import get_network

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch

BATCH_SIZE = 128
EPOCHS = 200
FIND_BASELINE = False  # used in validation to find a baseline that fully fits our training data
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
RATIOS = [1.5, 2, 2.5, 3]
defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def train(network, train_data, val_data, optimizer, scheduler, criterion, device, writer, path):
    curr_best_accuracy = 0
    current_state_dict = None
    best_accuracy_epoch = 0
    step = 0
    network.train()
    for epoch in range(1, EPOCHS + 1):
        current_loss = 0
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} of {EPOCHS}")
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
        curr_accuracy = val(network, val_data, device)
        network.train()
        if curr_accuracy > curr_best_accuracy:
            curr_best_accuracy = curr_accuracy
            current_state_dict = network.state_dict()
            best_accuracy_epoch = epoch
        print(f'Current accuracy: {curr_accuracy}')
        print(f'Loss: {current_loss.item()}')
        print(f'LR: {curr_lr}')
        writer.add_scalar('Training Loss', current_loss.item(), global_step=step)
        writer.add_scalar('Test accuracy', curr_accuracy, global_step=step)
        step += 1
        writer.flush()
        print('--------------------------------------------------')
    print(f'Best accuracy was {curr_best_accuracy} at epoch {best_accuracy_epoch}')
    torch.save(current_state_dict, path)
    writer.close()


def val(network, val_data, device):
    correct = 0
    total = len(val_data.dataset)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_data):
            data, targets = data.to(device), targets.to(device)
            out = network(data)
            _, preds = out.max(1)
            correct += preds.eq(targets).sum()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ratio', type=float, default=1)
    args = parser.parse_args()

    current_ratio = args.ratio
    current_cfg = defaultcfg[11]
    for i in range(len(current_cfg)):
        if isinstance(current_cfg[i], int):
            current_cfg[i] = int(current_ratio * current_cfg[i])
    print(f'Current VGG11 config being used: {current_cfg} (ratio {current_ratio}x)')
    saved_file_name = f'vgg11_{current_ratio}x.pt'  # TODO change this later
    PATH = os.getcwd() + f'/Models/SavedModels/{saved_file_name}'
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f'runs/CIFAR100/VGG/{saved_file_name}')

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

    net = get_network('vgg11', 'cifar100', current_cfg)
    net.to(device)  # need to parallelize later

    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if FIND_BASELINE:
        train(net, trainloader, trainloader, optimizer, scheduler, criterion, device, writer, PATH)
    else:
        train(net, trainloader, testloader, optimizer, scheduler, criterion, device, writer, PATH)


if __name__ == '__main__':
    main()
