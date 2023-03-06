import os
import sys 
import shutil

import torch
import argparse
from datetime import datetime
from pathlib import Path


def set_seed(seed=0):
    print('set_seed({})'.format(seed))
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

def argument_setup():
    set_seed()
    default_opb = 2
    parser = argparse.ArgumentParser(description='base setup')

    # operation setting
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='0', type=int, help='-1 = cpu')
    parser.add_argument('--cache_dir', type=str, default=Path('cache'))
    parser.add_argument('--check_dir', type=str, default=Path('ckpt'))
    parser.add_argument('--overwrite_opt', type=str, choices=['assert_fresh', 'overwrite'],default='overwrite') # move_to_trash

    # high level experiment decisions
    parser.add_argument('--model', type=str, choices=['consis'], default='consis') 
    parser.add_argument('--dataset', '--d', type=str, choices=['ciao', 'epinions', 'library'], default='epinions')
    parser.add_argument('--nop', type=int, choices=[1,2,3,4,5], default=2) #number_of_players
    parser.add_argument('--task', type=str, default='mca')
    parser.add_argument('--method', type=str, default='none')  
    parser.add_argument('--task_dir', type=str, default=None)
    parser.add_argument('--demographic_path', type=str, default=None)
    parser.add_argument('--num_proxy_items', type=int, default=100)
    parser.add_argument('--adv_rounds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--pretrain_epochs', type=int, default=0) 
    parser.add_argument('--recsys_epochs', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--target_rating', type=int, default=5)
    parser.add_argument('--compete_coeff', type=float, default=0.3)
    parser.add_argument('--overlap', type=float, default=1)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--percent', type=float, default=0.6, help='neighbor percent')
    parser.add_argument('--reg', type=float, default=1)
    parser.add_argument('--blist', type=str, default="1 2 3 4 5")
    parser.add_argument('--opb', type=int, default=default_opb)

    args, __ = parser.parse_known_args()

    # Setups
    ## Setup conda device
    args.device = torch.device("cpu" if args.device == -1 else "cuda:"+str(args.device))
    print('[setting] using device', args.device)

    ## Setup Path version of the directories, make directories
    args.check_dir, args.cache_dir = map(Path, [args.check_dir, args.cache_dir])
    args.task_method = args.task if args.task == 'none' else args.task + '_' + args.method

    ## budget
    args.blist = [int(i) for i in args.blist.split()]
    args.demographic_path = Path('demo')/(args.dataset+'.pt')
    args.default_opb = default_opb
    if args.task_dir is None:
        # nop, dataset, task, method determine poison; apply to models
        opcap = '' if args.opb == default_opb else str(args.opb)
        args.task_dir = args.check_dir / (str(args.nop)+'p'+opcap)/ '_'.join([args.dataset, args.task_method]) 
        if os.path.exists(args.task_dir):
            if args.overwrite_opt == 'assert_fresh':
                print('not a fresh start, exiting')
                sys.exit()
            elif args.overwrite_opt == 'overwrite':
                print('overwriting existing directory')
                # shutil.rmtree(args.task_dir)
            else:
                raise Exception(args.overwrite_opt + " not a legitamite overwrite option")
        else:
            os.makedirs(args.task_dir)
    else: 
        print('Conducting special experiment settings')
        print('assert domains are same as nop')
        args.task_dir = Path(args.task_dir)
    print_brief(parser, args)
    return args

class default_args():
    def __init__(self):
        self.cache_dir = Path('cache')
        self.overlap = 1
        self.embed_dim = 8
        self.percent = 0.6 
        self.reg = 1
        self.device = 0
        self.recsys_epochs = 100

def print_brief(parser, args, save=True):
    message = ''
    message += '----------------- Options ---------------\n'
    for k in ['task_dir', 'model', 'blist', 'opb']:
        v = vars(args)[k]
        default = parser.get_default(k)
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)


if __name__ == '__main__':
    args = argument_setup()
