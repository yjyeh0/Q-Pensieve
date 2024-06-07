import os
import argparse
from datetime import datetime
import gym
import torch
import numpy as np
import random

# import d4rl  # Import required to register environments, you may need to also import the submodule

# from environments.dst_d import DeepSeaTreasure
# from environments.MO_lunar_lander5d import LunarLanderContinuous
# from environments import hopper_v3, hopper5d_v3, half_cheetah_v3, ant_v3, walker2d_v3, hopper3d_v3, ant3d_v3
import environments
from agent import SacAgent
import os

import pickle
from state_norm_params import state_norm_params # we use normalization parameter for states from the behavioral policy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_id', type=str, default='dst_d-v0')
    parser.add_argument('--env_id', type=str, default='MO-Hopper-v2')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefer', type=int, default=4)
    parser.add_argument('--buf_num', type=int, default=0)
    parser.add_argument('--q_freq', type=int, default=1000)

    parser.add_argument('--dataset', type=str, nargs='+', default=['amateur_uniform'])
    '''
    parser.add_argument('--normalize_reward', type=bool, default=False)
    parser.add_argument('--dir', type=str, default='test_dir')
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=5000)
    '''

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
        'eval_interval': 5000, #50000,
        'cuda': args.cuda,
        'seed': args.seed,
        'cuda_device': args.cuda_device,
        'q_frequency': args.q_freq,
        'model_saved_step': 100000
    }

    dataset_paths = [f"data/{args.env_id}/{args.env_id}_50000_{d}.pkl" for d in args.dataset]
    trajectories = []
    for data_path in dataset_paths:
        with open(data_path, 'rb') as f:
            trajectories.extend(pickle.load(f))

    '''
    states, traj_lens, returns, returns_mo, preferences = [], [], [], [], []
    min_each_obj_step = np.min(np.vstack([np.min(traj['raw_rewards'], axis=0) for traj in trajectories]), axis=0)
    max_each_obj_step = np.max(np.vstack([np.max(traj['raw_rewards'], axis=0) for traj in trajectories]), axis=0)

    for traj in trajectories:
        # if concat_state_pref != 0:
        #     traj['observations'] = np.concatenate(
        #         (traj['observations'], np.tile(traj['preference'], concat_state_pref)), axis=1)

        if args.normalize_reward:
            traj['raw_rewards'] = (traj['raw_rewards'] - min_each_obj_step) / (max_each_obj_step - min_each_obj_step)

        traj['rewards'] = np.sum(np.multiply(traj['raw_rewards'], traj['preference']), axis=1)
        states.append(traj['observations'])
        traj_lens.append(len(traj['observations']))
        returns.append(traj['rewards'].sum())
        returns_mo.append(traj['raw_rewards'].sum(axis=0))
        preferences.append(traj['preference'][0, :])

    traj_lens, returns, returns_mo, states, preferences = np.array(traj_lens), np.array(returns), np.array(
        returns_mo), np.array(states), np.array(preferences)

    # if not isCloseToOne(percent_dt):
    #     num_traj_wanted = int(percent_dt * len(trajectories))
    #     indices_wanted = np.unique(np.argpartition(returns_mo, -num_traj_wanted, axis=0)[-num_traj_wanted:])
    #     trajectories = np.array([trajectories[i] for i in indices_wanted])
    #     traj_lens = traj_lens[indices_wanted]
    #     returns = returns[indices_wanted]
    #     returns_mo = returns_mo[indices_wanted, :]
    #     states = states[indices_wanted]
    #     preferences = preferences[indices_wanted, :]

    # states = np.concatenate(states, axis=0)
    # state_mean = f[env_id]["mean"]
    # state_std = np.sqrt(state_norm_params[env_id]["var"])
    # state_mean = np.concatenate((state_mean, np.zeros(concat_state_pref * pref_dim)))
    # state_std = np.concatenate((state_std, np.ones(concat_state_pref * pref_dim)))
    # state_dim += pref_dim * concat_state_pref

    '''
    env = gym.make(args.env_id)
    
    log_dir = os.path.join(
        'logs', args.env_id,
        f'MOSAC-set{args.prefer}-buf{args.buf_num}-seed{args.seed}_freq{args.q_freq}')

    agent = SacAgent(env=env, log_dir=log_dir, **configs)
    # agent.load_dataset_to_memory(trajectories)
    # agent.run()
    num_step_to_learn = int(len(trajectories) * 500 / configs['num_steps'])
    agent.run_offline(trajectories, num_step_to_learn)


if __name__ == '__main__':


    # Create the environment
    # env = gym.make('hopper-expert-v0')
    #
    # # d4rl abides by the OpenAI gym interface
    # env.reset()
    # env.step(env.action_space.sample())
    #
    # # Each task is associated with a dataset
    # # dataset contains observations, actions, rewards, terminals, and infos
    # dataset = env.get_dataset()
    # print(dataset['observations'])  # An N x dim_observation Numpy array of observations
    #
    # # Alternatively, use d4rl.qlearning_dataset which
    # # also adds next_observations.
    # dataset = d4rl.qlearning_dataset(env)
    run()
