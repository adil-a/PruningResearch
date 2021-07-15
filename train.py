import torch


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
    print(f'Best accuracy was {curr_best_accuracy} at epoch {best_accuracy_epoch}')
    torch.save(network.state_dict(), path_final_epoch)
    writer.close()


def eval(network, val_data, device, criterion, validate=False, validate_amount=0.1):
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
