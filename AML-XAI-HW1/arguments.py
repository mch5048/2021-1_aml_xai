import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual Learning')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    # CUB: 0.005
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1. Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (L2 penalty).')
    parser.add_argument('--lamb', type=float, default=1, help='Lambda for ewc')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--nepochs', type=int, default=60, help='Number of epochs for each increment')
    parser.add_argument('--tasknum', default=5, type=int, help='(default=%(default)s)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--dataset', default='CIFAR100', type=str,
                        choices=['CIFAR100', 'MNIST'], 
                        help='(default=%(default)s)')
    
    parser.add_argument('--trainer', default='ewc', type=str,
                        choices=['ewc', 'l2', 'vanilla'], 
                        help='(default=%(default)s)')
    
    args = parser.parse_args()
    return args
