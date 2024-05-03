import os
import argparse
from datetime import datetime
import gym
import torch
import numpy as np
import random

import d4rl  # Import required to register environments, you may need to also import the submodule

from environments.dst_d import DeepSeaTreasure
from environments.MO_lunar_lander5d import LunarLanderContinuous
from environments import hopper_v3, hopper5d_v3, half_cheetah_v3, ant_v3, walker2d_v3, hopper3d_v3, ant3d_v3
from agent import SacAgent
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='dst_d-v0')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefer', type=int, default=4)
    parser.add_argument('--buf_num', type=int, default=0)
    parser.add_argument('--q_freq', type=int, default=1000)
    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': 1500000,
        'batch_size': 256,#256
        'lr': 0.0003,
        'hidden_units': [256, 256],
        'memory_size': 1e6,
        'prefer_num': args.prefer,
        'buf_num': args.buf_num,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': 1,
        'per': False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': 50000,
        'cuda': args.cuda,
        'seed': args.seed,
        'cuda_device': args.cuda_device,
        'q_frequency': args.q_freq,
        'model_saved_step': 100000
    }
    
    env = gym.make(args.env_id)
    
    log_dir = os.path.join(
        'logs', args.env_id,
        f'MOSAC-set{args.prefer}-buf{args.buf_num}-seed{args.seed}_freq{args.q_freq}')

    agent = SacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    # Create the environment
    env = gym.make('hopper-expert-v0')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset['observations'])  # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    dataset = d4rl.qlearning_dataset(env)
    run()
