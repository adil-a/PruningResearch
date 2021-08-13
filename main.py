# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/main.py

import argparse
import os

from Pruners import singleshot
from Utils import config
from Plotters import pruning_inference, inference_graph
import OverparameterizationVerification
from Pruners.IMP import finetuning
from Pruners import imp_singleshot_mask_mix

if __name__ == '__main__':
    config.setup_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument('--graph', type=str, choices=['num_of_params', 'pruned_accuracies', 'weights_per_layer',
                                                      'unpruned_accuracies', 'mask_mix', 'singleshot_imp'])
    parser.add_argument('--overparameterization-verification', type=bool)
    parser.add_argument('--imp', type=bool)
    parser.add_argument('--imp-singleshot-mask-mix', type=bool)
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='mnist',
                               choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'],
                               help='dataset (default: mnist)')
    training_args.add_argument('--model_name', type=str, default='fc', choices=['fc', 'conv',
                                                                                'vgg11', 'vgg11-bn', 'vgg13',
                                                                                'vgg13-bn',
                                                                                'vgg16', 'vgg16-bn', 'vgg19',
                                                                                'vgg19-bn',
                                                                                'resnet18', 'resnet20', 'resnet32',
                                                                                'resnet34', 'resnet44', 'resnet50',
                                                                                'resnet56', 'resnet101', 'resnet110',
                                                                                'resnet110', 'resnet152', 'resnet1202',
                                                                                'wide-resnet18', 'wide-resnet20',
                                                                                'wide-resnet32', 'wide-resnet34',
                                                                                'wide-resnet44', 'wide-resnet50',
                                                                                'wide-resnet56', 'wide-resnet101',
                                                                                'wide-resnet110', 'wide-resnet110',
                                                                                'wide-resnet152', 'wide-resnet1202'],
                               help='model architecture (default: fc)')
    training_args.add_argument('--train-batch-size', type=int, default=64,
                               help='input batch size for training (default: 64)')
    training_args.add_argument('--test-batch-size', type=int, default=256,
                               help='input batch size for testing (default: 256)')
    training_args.add_argument('--post-epochs', type=int, default=10,
                               help='number of epochs to train after pruning (default: 10)')
    training_args.add_argument('--lr', type=float, default=0.001,
                               help='learning rate (default: 0.001)')
    training_args.add_argument('--expansion_ratio', type=float, default=1.0)
    training_args.add_argument('--lr-drops', type=int, nargs='*', default=[75, 150],
                               help='list of learning rate drops (default: [75, 150])')
    # Pruning Hyperparameters
    pruning_args = parser.add_argument_group('pruning')
    pruning_args.add_argument('--pruner', type=str, default='rand',
                              choices=['rand', 'mag', 'snip', 'grasp', 'synflow'],
                              help='prune strategy (default: rand)')
    pruning_args.add_argument('--compression', type=float, default=0.0,
                              help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
    pruning_args.add_argument('--prune-epochs', type=int, default=1,
                              help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--compression-schedule', type=str, default='exponential',
                              choices=['linear', 'exponential'],
                              help='whether to use a linear or exponential compression schedule (default: exponential)')
    pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global', 'local'],
                              help='masking scope (global or layer) (default: global)')
    pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                              help='ratio of prune dataset size and number of classes (default: 10)')
    pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                              help='input batch size for pruning (default: 256)')
    pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                              help='whether to prune in train mode (default: False)')
    pruning_args.add_argument('--reinitialize', type=bool, default=False,
                              help='IMP with reinitialization (default: False)')
    pruning_args.add_argument('--weight-rewind', type=bool, default=False,
                              help='Rewind weights after pruning (default: False)')
    pruning_args.add_argument('--imp-singleshot', type=bool, default=False,
                              help='IMP singleshot pruning (default: False)')
    pruning_args.add_argument('--shuffle', type=bool, default=False,
                              help='Shuffling masks (default: False)')
    # Experiment Hyperparameters
    parser.add_argument('--experiment', type=str, default='singleshot',
                        choices=['singleshot', 'multishot', 'unit-conservation',
                                 'layer-conservation', 'imp-conservation', 'schedule-conservation'],
                        help='experiment name (default: example)')
    args = parser.parse_args()
    if args.checkpoint_dir == '':
        args.checkpoint_dir = os.getcwd() + '/'
    if args.imp_singleshot_mask_mix:
        imp_singleshot_mask_mix.run(args)
    elif args.imp is True:
        finetuning.main(args)
    elif args.graph == 'unpruned_accuracies':
        inference_graph.main()
    elif args.graph is not None:
        pruning_inference.main(args)
    elif args.overparameterization_verification:
        OverparameterizationVerification.main(args)
    elif args.experiment == 'singleshot':
        singleshot.run(args)
