# this is still a very sandbox version of what the final file should look like

from Utils.Network_Retrieval import get_network

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch

BATCH_SIZE = 256
EPOCHS = 150


def train(network, train_data, val_data, optimizer, criterion, device, writer):
    step = 0
    network.train()
    for epoch in range(EPOCHS):
        current_loss = 0
        print(f"Epoch {epoch + 1} of {EPOCHS}")
        for batch_idx, (data, targets) in enumerate(train_data):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = network(data)
            loss = criterion(scores, targets)
            current_loss += loss
            loss.backward()
            optimizer.step()
        network.eval()
        curr_accuracy = val(network, val_data, device)
        print(f'Current accuracy: {curr_accuracy}')
        network.train()
        print(f'Loss: {current_loss.item()}')
        writer.add_scalar('Training Loss', current_loss.item(), global_step=step)
        writer.add_scalar('Test accuracy', curr_accuracy, global_step=step)
        step += 1
        writer.flush()
    writer.close()


def val(network, val_data, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_data):
            data, targets = data.to(device), targets.to(device)
            out = F.softmax(network(data), dim=1)
            total += targets.size()[0]
            out = out.argmax(dim=1)
            for idx in range(targets.size()[0]):
                if out[idx].item() == targets[idx].item():
                    correct += 1
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/CIFAR10/VGG/vgg19_run2')

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32)]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    net = get_network('vgg19')
    net.to(device)  # need to parallelize later

    optimizer = optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train(net, trainloader, testloader, optimizer, criterion, device, writer)


if __name__ == '__main__':
    main()
