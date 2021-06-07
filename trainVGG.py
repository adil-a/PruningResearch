# this is still a very sandbox version of what the final file should look like

from Models import VGGModels

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# import torch.nn.parallel
import torch

BATCH_SIZE = 256
EPOCHS = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

net = VGGModels.VGG()
net.to(device)  # need to parallelize later

optimizer = optim.Adam(net.parameters())
criterion = torch.nn.CrossEntropyLoss().to(device)


def train(network, train_data, optimizer, criterion):
    for epoch in range(EPOCHS):
        current_loss = 0
        print(f"Epoch {epoch + 1} of {EPOCHS}")
        for batch_idx, (data, targets) in enumerate(train_data):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = network(data)
            loss = criterion(scores, targets)
            current_loss += loss
            # print(loss)
            loss.backward()
            optimizer.step()
        print(f'Loss: {current_loss.item()}')


if __name__ == '__main__':
    train(net, trainloader, optimizer, criterion)
