# this is still a very sandbox version of what the final file should look like

from Models import VGGModels

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# import torch.nn.parallel
import torch

BATCH_SIZE = 256
EPOCHS = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/CIFAR10/VGG/vgg19_run1')

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
# valset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                       download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=test_transform)

# NUM_TRAIN = len(trainset)
# NUM_VAL = 1000
# indices = list(range(NUM_TRAIN))
# train_idx, val_idx = indices[NUM_VAL:], indices[:NUM_VAL]
# train_sampler = SubsetRandomSampler(train_idx)
# val_sampler = SubsetRandomSampler(val_idx)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)
# valloader = DataLoader(trainset, batch_size=BATCH_SIZE,
#                        num_workers=4, sampler=val_sampler)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

net = VGGModels.VGG()
net.to(device)  # need to parallelize later

optimizer = optim.Adam(net.parameters())
criterion = torch.nn.CrossEntropyLoss().to(device)


def train(network, train_data, val_data, optimizer, criterion):
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
        curr_accuracy = val(network, val_data)
        print(f'Current accuracy: {curr_accuracy}')
        network.train()
        print(f'Loss: {current_loss.item()}')
        writer.add_scalar('Training Loss', current_loss.item(), global_step=step)
        writer.add_scalar('Test accuracy', curr_accuracy, global_step=step)
        step += 1
        writer.flush()
    writer.close()


def val(network, val_data):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_data):
            data, targets = data.to(device), targets.to(device)
            out = F.softmax(network(data), dim=1)
            total += targets.size()[0]
            # torch.set_printoptions(edgeitems=5) # here for sanity check
            out = out.argmax(dim=1)
            # torch.set_printoptions(edgeitems=3)
            for idx in range(targets.size()[0]):
                if out[idx].item() == targets[idx].item():
                    correct += 1
    return correct / total


if __name__ == '__main__':
    train(net, trainloader, testloader, optimizer, criterion)
