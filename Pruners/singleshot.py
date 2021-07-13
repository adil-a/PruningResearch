# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/Experiments/singleshot.py

import numpy as np
import torch.nn as nn
import torch
from Utils import network_utils, pruning_utils, config
from prune import prune_loop


def run(args):
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Loading {args.dataset} dataset')
    input_shape, num_classes = network_utils.dimension(args.dataset)
    prune_loader = network_utils.dataloader(args.dataset, args.prune_batch_size, True,
                                            length=args.prune_dataset_ratio * num_classes)
    train_loader = network_utils.dataloader(args.dataset, args.train_batch_size, True)
    test_loader = network_utils.dataloader(args.dataset, args.test_batch_size, False)

    cfg = config.defaultcfg[11]
    network_utils.multiplier(cfg, args.expansion_ratio)
    model = network_utils.get_network(args.model_name, args.dataset, cfg, imp=False).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)

    print(f'Pruning with {args.pruner} for {args.prune_epochs} epochs')
    pruner = pruning_utils.pruner(args.pruner)(pruning_utils.masked_parameters(model))
    _, total_elems = pruner.stats()
    target_sparsity = 1 - (config.TARGET_SIZE / total_elems)
    prune_loop(model, loss, pruner, prune_loader, device, target_sparsity, args.compression_schedule, 'global',
               args.prune_epochs)

    # TODO implement training and main
