import os.path

import torch
from Utils.network_utils import eval, checkpointing


def train(network, train_data, val_data, optimizer, scheduler, criterion, device, writer, path, path_final_epoch,
          epochs, checkpoint_dir):
    epoch = 1
    curr_best_accuracy = 0
    best_accuracy_epoch = 0
    step = 0
    checkpoint_location = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_location):
        checkpoint = torch.load(checkpoint_location)
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = checkpoint['loss']
        epoch = checkpoint['epoch']
        if checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        rng = checkpoint['rng']
        torch.set_rng_state(rng)
        curr_best_accuracy = checkpoint['curr_best_accuracy']
        best_accuracy_epoch = checkpoint['best_accuracy_epoch']
        step = checkpoint['tb_step']
    network.train()
    while epoch < epochs + 1:
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
        if scheduler is not None:
            scheduler.step()
        network.eval()
        curr_test_accuracy, curr_test_loss = eval(network, val_data, device, criterion)
        curr_accuracy, _ = eval(network, train_data, device, criterion)
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
        epoch += 1
        if epoch % 10 == 0:
            checkpointing(network, optimizer, scheduler, criterion, epoch, torch.get_rng_state(), curr_best_accuracy,
                          best_accuracy_epoch, step, checkpoint_dir)
    print(f'Best accuracy was {curr_best_accuracy} at epoch {best_accuracy_epoch}')
    torch.save(network.state_dict(), path_final_epoch)
    writer.close()


def train_imp(network, train_data, test_data, epochs, optimizer, criterion, scheduler,
              device, path, file_name, pruning_iteration, writer, checkpoint_dir):
    network.train()
    epoch = 1
    step = 0
    curr_best_accuracy = 0
    curr_best_epoch = 0
    while epoch < epochs + 1:
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
            # for buffer_name, buffer in network.named_buffers():
            #     print((buffer.size(), buffer_name, buffer))
        if scheduler is not None:
            scheduler.step()
        network.eval()
        curr_test_accuracy, curr_test_loss = eval(network, test_data, device, criterion)
        curr_training_accuracy, _ = eval(network, train_data, device, criterion)
        if curr_test_accuracy > curr_best_accuracy:
            curr_best_accuracy = curr_test_accuracy
            curr_best_epoch = epoch
            model_save_path = path + file_name + f'_{pruning_iteration}_best.pt'
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
        epoch += 1
        # if epoch % 10 == 0:
        #     checkpointing(network, optimizer, scheduler, criterion, epoch, torch.get_rng_state(), curr_best_accuracy,
        #                   curr_best_epoch, step, checkpoint_dir)
    print(f'Best accuracy was {curr_best_accuracy} at epoch {curr_best_epoch}')
