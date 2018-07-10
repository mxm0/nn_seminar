from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from network import ActorCriticFFNetwork
from test import test
from train import train
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GAMMA
from constants import ENTROPY_BETA
from constants import GRAD_NORM_CLIP
from constants import TASK_LIST
from constants import ACTION_SIZE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR

# Credit
# https://github.com/ikostrikov/pytorch-a3c
# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0007,
                    help='learning rate (default: 0.0007)')
parser.add_argument('--gamma', type=float, default=GAMMA,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--alpha', type=float, default=RMSP_ALPHA,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--eps', type=float, default=RMSP_EPSILON,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=ENTROPY_BETA,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=GRAD_NORM_CLIP,
                    help='value loss coefficient (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=MAX_TIME_STEP,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='bathroom_02',
                    help='environment to train on (default: bathroom_02)')
parser.add_argument('--no_shared', default=False,
                    help='environment to train on (default: bathroom_02)')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    args = parser.parse_args()

    
    torch.manual_seed(args.seed)
    #env = create_atari_env(args.env_name)
    shared_model = ActorCriticFFNetwork(ACTION_SIZE)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha = args.alpha, eps = args.eps)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    '''
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)
    '''
    branches = []
    for scene in scene_scopes:
        for task in list_of_tasks[scene]:
            branches.append((scene, task))
    NUM_TASKS = len(branches)
    
    if os.path.exists(CHECKPOINT_DIR + '/' + 'checkpoint.pth.tar'):
        checkpoint = torch.load(CHECKPOINT_DIR + '/' + 'checkpoint.pth.tar',
                                map_location = lambda storage,
                                loc: storage)
        # set global step
        shared_model.load_state_dict(checkpoint)
        print("Mode loaded")
    else:
        print("Could not find old checkpoint")

    for rank in range(0, args.num_processes):
        scene, task = branches[rank%NUM_TASKS]
        p = mp.Process(target=train, args=(rank ,scene, task, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print('Now saving data. Please wait.')
    torch.save(shared_model.state_dict(),
                CHECKPOINT_DIR + '/' + 'checkpoint.pth.tar')
