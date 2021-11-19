import torch
from Models.TeacherStudentModels import MLPStudent, MLPTeacher, Model
from torch.utils.data.dataloader import DataLoader, Dataset
from Pruners.imp_singleshot_mask_mix import mask_swap
from torchvision import datasets, transforms
from Utils import pruning_utils
from Layers import layers
import argparse
from scipy.spatial.distance import cdist
from Utils.config import setup_seed
import os
from copy import deepcopy
import wandb

STUDENT_TRAINING_EPOCHS = 100
TEACHER_TRAINING_EPOCHS = 60
LR = 0.01
PRUNING_RATIO = 0.2
FINAL_SPARSITY = 0.1


class RandomDataset(Dataset):
    def __init__(self, N, d, std):
        super(RandomDataset, self).__init__()
        self.d = d
        self.std = std
        self.N = N
        self.regenerate()

    def regenerate(self):
        self.x = torch.FloatTensor(self.N, *self.d).normal_(0, std=self.std)

    def __getitem__(self, idx):
        return self.x[idx], -1

    def __len__(self):
        return self.N


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


def init_dataset(args):
    d = (args.data_d,)
    d_output = 100
    train_dataset = RandomDataset(100000, d, args.data_std)
    eval_dataset = RandomDataset(1024, d, args.data_std)

    return d, d_output, train_dataset, eval_dataset


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


def getCorrelations(teacher_network, student_network, prune=False):
    tn_activations = torch.cat(teacher_network.activations)
    sn_activations_total = torch.cat(student_network.activations)

    unpruned_nodes_activations = []
    # unpruned_nodes = student_network.hidden_layer1.bias_mask
    unpruned_nodes = student_network.ws_linear[0].bias_mask
    for i in range(unpruned_nodes.size(0)):  # bias mask to easily see unpruned nodes
        if unpruned_nodes[i] == 1:
            unpruned_nodes_activations.append(sn_activations_total[:, i].reshape(-1, 1))
    sn_activations = torch.hstack(unpruned_nodes_activations)
    tn_activations_cpu = tn_activations.T.cpu()
    sn_activations_cpu = sn_activations.T.cpu()

    correlation_matrix = 1 - torch.tensor(cdist(tn_activations_cpu, sn_activations_cpu, 'cosine'))
    maximum = torch.max(correlation_matrix, dim=1)[0]
    average = torch.mean(maximum)
    if prune:
        return average.item(), torch.argmax(correlation_matrix, dim=1, keepdim=True)
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


def optimize(train_loader, eval_loader, teacher, student, loss_func, active_nodes, pruner, pruning_iterations,
             initial_size, device, initial_student_network, args):
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr[0], momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.normalize:
        student.normalize()

    step = 0
    for i in range(args.num_epoch):
        total_loss = 0
        teacher.eval()
        student.train()
        if i in args.lr:
            lr = args.lr[i]
            print(f"[{i}]: lr = {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for x, y in train_loader:
            optimizer.zero_grad()
            if not args.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            output_t = teacher(x)
            output_s = student(x)

            err = loss_func(output_s["y"], output_t["y"].detach())
            # print(err.item())
            total_loss += err.item()
            err.backward()
            optimizer.step()
            if args.normalize:
                student.normalize()
        if i == 4 or i == 9:
            torch.save(student.state_dict(), f'student_network_epoch{i + 1}.pt')
        teacher.reset_activations()
        student.reset_activations()
        for x, y in eval_loader:
            if not args.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            teacher(x)
            student(x)
        corr = getCorrelations(teacher, student)
        test_error = gauss_test(teacher, student, eval_loader, device)
        print(f'Epoch: {i + 1}, train loss: {total_loss / len(train_loader)}')
        print(f'Mean corr: {corr}')
        print(f'Test loss: {test_error}')
        print('--------------------------------------------------------')
        wandb.log({'Correlation': corr, 'Train/Loss': total_loss / len(train_loader), 'Test/Loss': test_error},
                  step=step)
        step += 1
        if args.regen_dataset_each_epoch:
            train_loader.dataset.regenerate()

    # if teacher.ws_linear[0].weight.shape[0] >= student.ws_linear[0].weight.shape[0]:
    #     print("Pruning can't be done since student network's num nodes <= teacher network's num nodes")
    #     return
    # print('Removing non-specialized nodes')
    # _, specialized_nodes = getCorrelations(teacher, student, True)
    # pruner.score(student, None, None, None)
    # # don't prune specialized nodes
    # intermediate_masks = torch.zeros((student.ws_linear[0].weight.shape[0], 1)).to(device)
    # intermediate_masks[torch.flatten(specialized_nodes)] = 1
    # pruner.intermediate_masks = intermediate_masks
    # pruner.mask(None, 'global', specialized_ts=True)
    # node_count_vector = student.ws_linear[0].weight_mask[:, 0]
    # print(f'Node count after pruning {torch.sum(node_count_vector)} / {node_count_vector.shape[0]}')
    #
    # if args.reinitialize:
    #     initial_student_network.load_state_dict(torch.load('student_network_epoch10.pt'))
    #     mask_swap(initial_student_network, student)
    #     print('Weights reinitialized to epoch 10')
    #     student = initial_student_network
    #
    # for i in range(args.num_epoch):
    #     total_loss = 0
    #     teacher.eval()
    #     student.train()
    #     if i in args.lr:
    #         lr = args.lr[i]
    #         print(f"[{i}]: lr = {lr}")
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    #     for x, y in train_loader:
    #         optimizer.zero_grad()
    #         if not args.use_cnn:
    #             x = x.view(x.size(0), -1)
    #         x = x.cuda()
    #         output_t = teacher(x)
    #         output_s = student(x)
    #
    #         err = loss_func(output_s["y"], output_t["y"].detach())
    #         total_loss += err.item()
    #         err.backward()
    #         optimizer.step()
    #         if args.normalize:
    #             student.normalize()
    #
    #     teacher.reset_activations()
    #     student.reset_activations()
    #     for x, y in eval_loader:
    #         if not args.use_cnn:
    #             x = x.view(x.size(0), -1)
    #         x = x.cuda()
    #         teacher(x)
    #         student(x)
    #     corr = getCorrelations(teacher, student)
    #     test_error = gauss_test(teacher, student, eval_loader, device)
    #     print(f'Epoch: {i + 1}, train loss: {total_loss / len(train_loader)}')
    #     print(f'Mean corr: {corr}')
    #     print(f'Test loss: {test_error}')
    #     print('--------------------------------------------------------')
    #     wandb.log({'Correlation': corr, 'Train/Loss': total_loss / len(train_loader), 'Test/Loss': test_error},
    #               step=step)
    #     step += 1

    # print(f'Total pruning iterations: {pruning_iterations}')
    # for pruning_iteration in range(1, pruning_iterations + 1):
    #     sparsity = 0.03 ** (pruning_iteration / pruning_iterations)
    #     print(f'Pruning to sparsity {sparsity}')
    #     pruner.score(None, None, None, None)
    #     pruner.mask(sparsity, 'global')
    #     print(f'Current weights / total weights: {int(pruner.stats()[0])}/{initial_size}')
    #     for epoch in range(5):
    #         teacher.eval()
    #         student.train()
    #         # if i in args.lr:
    #         #     lr = args.lr[i]
    #         #     print(f"[{i}]: lr = {lr}")
    #         #     for param_group in optimizer.param_groups:
    #         #         param_group['lr'] = lr
    #         # sample data from Gaussian distribution.
    #         # xsel = Variable(X.gather(0, sel))
    #         for x, y in train_loader:
    #             optimizer.zero_grad()
    #             if not args.use_cnn:
    #                 x = x.view(x.size(0), -1)
    #             x = x.cuda()
    #             output_t = teacher(x)
    #             output_s = student(x)
    #
    #             err = loss_func(output_s["y"], output_t["y"].detach())
    #             err.backward()
    #             optimizer.step()
    #             if args.normalize:
    #                 student.normalize()
    #
    #         teacher.reset_activations()
    #         student.reset_activations()
    #         for x, y in eval_loader:
    #             if not args.use_cnn:
    #                 x = x.view(x.size(0), -1)
    #             x = x.cuda()
    #             teacher(x)
    #             student(x)
    #
    #         # print(getCorrelations(teacher, student))
    #         wandb.log({'Correlation': getCorrelations(teacher, student)}, step=step)
    #         step += 1


def gauss_test(teacher, student, testLoader, device):
    teacher.to(device)
    student.to(device)
    running_loss = 0
    with torch.no_grad():
        temp_loss = torch.nn.MSELoss()
        for batch_idx, (data, targets) in enumerate(testLoader):
            data, targets = data.to(device), targets.to(device)
            running_loss += temp_loss(teacher(data)['y'], student(data)['y'])
        return running_loss / len(testLoader)


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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_iter', type=int, default=30000)
    parser.add_argument('--node_multi', type=int, default=10)
    parser.add_argument('--init_multi', type=int, default=4)
    parser.add_argument("--lr", type=str, default="0.01")
    parser.add_argument("--data_d", type=int, default=20)
    parser.add_argument("--data_std", type=float, default=10.0)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--num_trial", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--eval_batchsize", type=int, default=64)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--json_output", action="store_true")
    parser.add_argument("--cross_entropy", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perturb", type=float, default=None)
    parser.add_argument("--same_dir", action="store_true")
    parser.add_argument("--same_sign", action="store_true")
    parser.add_argument("--normalize", action="store_true",
                        help="Whether we normalize the weight vector after each epoch")
    parser.add_argument("--dataset", choices=["mnist", "gaussian", "cifar10"], default="gaussian")
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--load_teacher", type=str, default=None)
    parser.add_argument("--d_output", type=int, default=0)
    parser.add_argument("--ks", type=str, default='[10, 15, 20, 25]')
    parser.add_argument("--bn_affine", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--no_sep", action="store_true")

    parser.add_argument("--teacher_bn_affine", action="store_true")
    parser.add_argument("--teacher_bn", action="store_true")

    parser.add_argument("--stats_H", action="store_true")
    parser.add_argument("--stats_w", action="store_true")
    parser.add_argument("--use_cnn", action="store_true")
    parser.add_argument("--bn_before_relu", action="store_true")
    parser.add_argument("--regen_dataset_each_epoch", action="store_true")
    parser.add_argument("--reinitialize", type=bool, default=False)
    parser.add_argument("--weight_init", type=bool, default=True)
    args = parser.parse_args()
    args.ks = eval(args.ks)
    args.lr = eval(args.lr)
    args.weight_init = True
    if not isinstance(args.lr, dict):
        args.lr = {0: args.lr}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.perturb is not None or args.same_dir or args.same_sign:
        args.node_multi = 1
    d, d_output, train_dataset, eval_dataset = init_dataset(args)
    if args.d_output > 0:
        d_output = args.d_output

    teacher = Model(d[0], args.ks, d_output,
                    has_bias=not args.no_bias, has_bn=args.teacher_bn, has_bn_affine=args.teacher_bn_affine,
                    bn_before_relu=args.bn_before_relu).cuda()

    wandb.login()
    configuration = dict(learning_rate=LR,
                         train_epochs=STUDENT_TRAINING_EPOCHS)
    wandb.init(project='Pruning Research',
               config=configuration,
               entity='sparsetraining')
    wandb.run.name = f'StudentNetwork_{args.node_multi * args.ks[0]}_nodes_{d[0]}_dimensionality_weights_' \
                     f'initialized_{args.weight_init}'
    wandb.run.save()
    if args.weight_init:
        print("Init teacher..`")
        teacher.init_w(use_sep=not args.no_sep)
        teacher.normalize()
        print("Teacher weights initiailzed randomly...")
    active_nodes = None
    active_ks = args.ks

    student = Model(d[0], active_ks, d_output,
                    multi=args.node_multi,
                    has_bias=not args.no_bias, has_bn=args.bn, has_bn_affine=args.bn_affine,
                    bn_before_relu=args.bn_before_relu).to(device)
    initial_student_network = None
    if args.reinitialize:
        initial_student_network = Model(d[0], active_ks, d_output,
                                        multi=args.node_multi,
                                        has_bias=not args.no_bias, has_bn=args.bn, has_bn_affine=args.bn_affine,
                                        bn_before_relu=args.bn_before_relu).to(device)
        torch.save(student.state_dict(), 'student_network_epoch0.pt')
    # for name, module in teacher.named_modules():
    #     if isinstance(module, layers.Linear):
    #         print(name)
    #         print(module)
    #
    # for name, module in student.named_modules():
    #     if isinstance(module, layers.Linear):
    #         print(name)
    #         print(module)

    loss = torch.nn.MSELoss().cuda()

    temp_pruner = pruning_utils.pruner('Mag')(pruning_utils.masked_parameters(teacher))
    teacher_size = temp_pruner.stats()[1]
    # student_pruner = pruning_utils.pruner('Mag')(pruning_utils.masked_parameters(student))
    student_pruner = pruning_utils.pruner('Specialized')(pruning_utils.masked_parameters(student))
    student_size = student_pruner.stats()[1]
    pruning_iters = pruning_utils.get_finetune_iterations(teacher_size, student_size, 0.2)


    def loss_func(y, target):
        return loss(y, target)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batchsize, shuffle=True, num_workers=4)

    optimize(train_loader, eval_loader, teacher, student, loss_func, active_nodes, student_pruner, pruning_iters,
             student_size, device, initial_student_network, args)

    # setup_seed(1)
    # TeacherModel = MLPTeacher(100, 784, 10)
    # StudentModel = MLPStudent(1000, 784, 10)
    # student_train_loader, student_test_loader = getStudentLoaders(TeacherModel, 10)
    # teacher_train_loader, teacher_test_loader = getTeacherLoaders(10)
    # device = torch.device('cuda')
    #
    # # train(TeacherModel, TEACHER_TRAINING_EPOCHS, 0.1, train_loader, test_loader, device)
    # # torch.save(TeacherModel.state_dict(), os.getcwd() + '/Models/SavedModels/TeacherNetwork.pt')
    # TeacherModel.load_state_dict(torch.load(os.getcwd() + '/Models/SavedModels/TeacherNetwork.pt'))
    # print(temp(TeacherModel, teacher_test_loader, device))
    #
    # # train(StudentModel, STUDENT_TRAINING_EPOCHS, 0.01, student_train_loader, student_test_loader, device)
    # # torch.save(StudentModel.state_dict(), os.getcwd() + '/Models/SavedModels/StudentNetwork.pt')
    # # StudentModel.load_state_dict(torch.load(os.getcwd() + '/Models/SavedModels/StudentNetwork.pt'))
    # StudentModel.to(device)
    # TeacherModel.to(device)
    # pruner = pruning_utils.pruner('Mag')(pruning_utils.masked_parameters(StudentModel))
    # pruning_iterations = pruning_utils.get_finetune_iterations(100, 1000, PRUNING_RATIO)
    # train(StudentModel, STUDENT_TRAINING_EPOCHS, 0.01, student_train_loader, student_test_loader, device,
    #       pruning_iterations, 50, pruner, TeacherModel)
    # torch.save(StudentModel.state_dict(), os.getcwd() + '/Models/SavedModels/StudentNetwork.pt')
