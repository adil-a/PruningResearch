import torch
from Models.TeacherStudentModels import MLPStudent, MLPTeacher
from torch.utils.data.dataloader import DataLoader, Dataset
from torchvision import datasets, transforms
from Utils import pruning_utils
from Layers import layers
from scipy.spatial.distance import cdist
from Utils.config import setup_seed
import os
import wandb

STUDENT_TRAINING_EPOCHS = 100
TEACHER_TRAINING_EPOCHS = 60
LR = 0.01
PRUNING_RATIO = 0.2
FINAL_SPARSITY = 0.1


class GaussDataset(Dataset):
    def __init__(self, input_dataset, labels):
        self.input = input_dataset
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        training_example = self.input[idx]
        return training_example, label


class MNISTStudent(Dataset):
    def __init__(self, input_dataset, labels):
        self.input = input_dataset
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        training_example = self.input[idx]
        return training_example, label


def getTeacherLoaders(num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, 128, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, 128, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def getStudentLoaders(trained_network, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    temp_train_set = datasets.MNIST('../data', train=True, download=True,
                                    transform=transform)
    temp_test_set = datasets.MNIST('../data', train=False,
                                   transform=transform)
    train_X = temp_train_set.data.float()
    test_X = temp_test_set.data.float()
    with torch.no_grad():
        train_y = torch.exp(trained_network(train_X.reshape(train_X.size(0), -1)))
        test_y = torch.exp(trained_network(test_X.reshape(test_X.size(0), -1)))
    train_set = MNISTStudent(train_X, train_y)
    test_set = MNISTStudent(test_X, test_y)
    train_loader = torch.utils.data.DataLoader(train_set, 128, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, 128, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def getCorrelations(teacher_network, student_network):
    tn_activations = torch.cat(teacher_network.activations)
    sn_activations_total = torch.cat(student_network.activations)

    unpruned_nodes_activations = []
    unpruned_nodes = student_network.hidden_layer1.bias_mask
    for i in range(unpruned_nodes.size(0)):  # bias mask to easily see unpruned nodes
        if unpruned_nodes[i] == 1:
            unpruned_nodes_activations.append(sn_activations_total[:, i].reshape(-1, 1))
    sn_activations = torch.hstack(unpruned_nodes_activations)
    # print(tn_activations.shape)
    # print(sn_activations.shape)
    # tn_mean = torch.mean(tn_activations, dim=0, keepdim=True)
    # tn_std = torch.std(tn_activations, dim=0, keepdim=True)
    # sn_mean = torch.mean(sn_activations, dim=0, keepdim=True)
    # sn_std = torch.std(sn_activations, dim=0, keepdim=True)
    # tn_activations.sub_(tn_mean)
    # tn_activations.div_(tn_std)
    # sn_activations.sub_(sn_mean)
    # sn_activations.div_(sn_std)
    # print(tn_activations)
    tn_activations_cpu = tn_activations.T.cpu()
    sn_activations_cpu = sn_activations.T.cpu()

    correlation_matrix = 1 - torch.tensor(cdist(tn_activations_cpu, sn_activations_cpu, 'cosine'))
    # correlation_matrix = tn_activations.T @ sn_activations
    # print(correlation_matrix)
    maximum = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
    # print(maximum)
    average = torch.mean(maximum)
    print(f'Mean corr: {average.item()}')
    return average.item()


def train(network, training_epochs, lr, trainLoader, testLoader, device, pruning_iters, pruning_epochs, pruner,
          teacher_network):
    curr_pruning_iter = 1
    network.to(device)
    teacher_network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr, weight_decay=5e-4)
    if isinstance(network, MLPTeacher):
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.MSELoss()
    criterion.to(device)
    total_epochs = training_epochs + pruning_iters * pruning_epochs
    print(f'Total epochs: {total_epochs}')
    for i in range(1, total_epochs + 1):
        total_loss = 0
        network.train()
        if i >= training_epochs and (i - training_epochs) % pruning_epochs == 0 and \
                curr_pruning_iter <= pruning_iters:
            sparsity = FINAL_SPARSITY ** (curr_pruning_iter / pruning_iters)
            curr_pruning_iter += 1
            pruner.score(network, None, None, device)
            pruner.mask(sparsity, 'global', True)
            print(f'Pruned to sparsity {sparsity}')
        for batch_idx, (data, targets) in enumerate(trainLoader):
            data = data.reshape(data.size(0), -1)
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = network(data)
            loss = criterion(scores, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        network.eval()
        correct_train = 0
        correct_test = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(trainLoader):
                data = data.reshape(data.size(0), -1)
                data, targets = data.to(device), targets.to(device)
                scores = network(data)
                if isinstance(network, MLPTeacher):
                    discrete_preds = torch.argmax(torch.exp(scores), dim=1)
                    correct_train += discrete_preds.eq(targets).sum()
                else:
                    discrete_preds = torch.argmax(scores, dim=1)
                    discrete_truths = torch.argmax(targets, dim=1)
                    correct_train += discrete_preds.eq(discrete_truths).sum()

        network.reset_activations()
        teacher_network.reset_activations()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(testLoader):
                data = data.reshape(data.size(0), -1)
                data, targets = data.to(device), targets.to(device)
                scores = network(data)
                test_loss += criterion(scores, targets).item()
                teacher_network(data)  # record activations
                if isinstance(network, MLPTeacher):
                    discrete_preds = torch.argmax(torch.exp(scores), dim=1)
                    correct_test += discrete_preds.eq(targets).sum()
                else:
                    discrete_preds = torch.argmax(scores, dim=1)
                    discrete_truths = torch.argmax(targets, dim=1)
                    correct_test += discrete_preds.eq(discrete_truths).sum()
        correlation = getCorrelations(teacher_network, network)

        print(f'Epoch {i}, loss: {total_loss / len(trainLoader)}, train accuracy: '
              f'{correct_train / len(trainLoader.dataset)}, test accuracy: '
              f'{correct_test / len(testLoader.dataset)}')
        wandb.log({'training/loss': total_loss / len(trainLoader),
                   'training/accuracy': correct_train / len(trainLoader.dataset)}, step=i)
        wandb.log({'test/loss': test_loss / len(testLoader), 'test/accuracy': correct_test / len(testLoader.dataset),
                   'test/correlation': correlation}, step=i)


# X_train = torch.normal(0, 5, (10000, 5))
# y_train = torch.normal(0, 5, (10000, 5))
# X_test = torch.normal(0, 5, (1000, 5))
# y_test = torch.normal(0, 5, (1000, 5))
# ds_train = GaussDataset(X_train, y_train)
# ds_test = GaussDataset(X_test, y_test)
# train_loader = DataLoader(ds_train, 256)
# test_loader = DataLoader(ds_test, 256)

def temp(network, testLoader, device):
    correct_test = 0
    network.to(device)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(testLoader):
            data = data.reshape(data.size(0), -1)
            data, targets = data.to(device), targets.to(device)
            scores = network(data)
            if isinstance(network, MLPTeacher):
                discrete_preds = torch.argmax(torch.exp(scores), dim=1)
                correct_test += discrete_preds.eq(targets).sum()
            else:
                discrete_preds = torch.argmax(scores, dim=1)
                discrete_truths = torch.argmax(targets, dim=1)
                correct_test += discrete_preds.eq(discrete_truths).sum()
    return correct_test / len(testLoader.dataset)


if __name__ == '__main__':
    wandb.login()
    configuration = dict(learning_rate=LR,
                         train_epochs=STUDENT_TRAINING_EPOCHS)
    wandb.init(project='Pruning Research',
               config=configuration,
               entity='sparsetraining')
    wandb.run.name = 'StudentNetwork'
    wandb.run.save()

    setup_seed(1)
    TeacherModel = MLPTeacher(100, 784, 10)
    StudentModel = MLPStudent(1000, 784, 10)
    student_train_loader, student_test_loader = getStudentLoaders(TeacherModel, 10)
    teacher_train_loader, teacher_test_loader = getTeacherLoaders(10)
    device = torch.device('cuda')

    # train(TeacherModel, TEACHER_TRAINING_EPOCHS, 0.1, train_loader, test_loader, device)
    # torch.save(TeacherModel.state_dict(), os.getcwd() + '/Models/SavedModels/TeacherNetwork.pt')
    TeacherModel.load_state_dict(torch.load(os.getcwd() + '/Models/SavedModels/TeacherNetwork.pt'))
    print(temp(TeacherModel, teacher_test_loader, device))

    # train(StudentModel, STUDENT_TRAINING_EPOCHS, 0.01, student_train_loader, student_test_loader, device)
    # torch.save(StudentModel.state_dict(), os.getcwd() + '/Models/SavedModels/StudentNetwork.pt')
    # StudentModel.load_state_dict(torch.load(os.getcwd() + '/Models/SavedModels/StudentNetwork.pt'))
    StudentModel.to(device)
    TeacherModel.to(device)
    pruner = pruning_utils.pruner('Mag')(pruning_utils.masked_parameters(StudentModel))
    pruning_iterations = pruning_utils.get_finetune_iterations(100, 1000, PRUNING_RATIO)
    train(StudentModel, STUDENT_TRAINING_EPOCHS, 0.01, student_train_loader, student_test_loader, device,
          pruning_iterations, 50, pruner, TeacherModel)
    torch.save(StudentModel.state_dict(), os.getcwd() + '/Models/SavedModels/StudentNetwork.pt')
