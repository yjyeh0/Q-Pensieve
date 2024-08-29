import os
import visdom
import numpy as np
import torch
import copy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# from rltorch.memory import MultiStepMemory, PrioritizedMemory
from base import QMemory

from model import MultiQNetwork, GaussianPolicy
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
import random
from multi_step import *
from prioritized import *
from datetime import datetime
import time

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pickle

writer = SummaryWriter()

PREF = [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]
p_name= ['9010','5050','1090']
#PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8],[0.1,0.9]]

# p_name= ['9505','9010','8515','8020','7525','7030','6535','6040','5545','5050','4555','4060','3565','3070','2575','2080','1585','1090','0595']
# PREF = [[0.95,0.05],[0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.75, 0.25], [0.7, 0.3], [0.65, 0.35], [0.6, 0.4], [0.55, 0.35], [0.5, 0.5], [0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7], [0.25, 0.75],[0.2, 0.8], [0.15,0.85] ,[0.1,0.9]]
class QMonitor(object):
    def __init__(self,train=True):
        a=1
    def update(self, eps, a, b, c, d):
        a=1
        
class Monitor(object):

    def __init__(self, spec,path):

        env_name = spec['env_name']
        set_num = spec['set_num']
        buf_num = spec['buf_num']
        self.vis = visdom.Visdom(env = f'MOSAC{datetime.now().strftime("%m%d")}-{env_name}_pref{set_num}_buf{buf_num}' ,port = 8097)
        self.spec = spec
        if spec['pref'][0] == 0.9:
            print(999)
            self.path = os.path.join(path,'reward_log91.npz')
        elif spec['pref'][0] == 0.5:
            print(555)
            self.path = os.path.join(path,'reward_log55.npz')
        elif spec['pref'][0] == 0.1:
            print(111)
            self.path = os.path.join(path,'reward_log19.npz')


        self.value_window = None
        self.text_window = None

    def update(self, eps, tot_reward, Rew_1, Rew_2, loss):

        if self.value_window == None:
            self.tot_t = np.array([tot_reward])
            self.rew_1_t = np.array([Rew_1])
            self.rew_2_t = np.array([Rew_2])
            self.loss_t = np.array([loss])
            self.value_window = self.vis.line(X=torch.Tensor([eps]).cpu(),
                                              Y=torch.Tensor([tot_reward, Rew_1, Rew_2, loss]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='steps_per10000',
                                                        ylabel='Reward value',
                                                        title='Value Dynamics ' + str(self.spec['pref']) + ' ' + str(self.spec['seed']),
                                                        legend=['Total Reward', 'forward_reward', 'ctrl cost','loss']))
        else:
            #Smoothing
            self.tot_t = np.append(self.tot_t, tot_reward)
            tot_reward = np.mean(self.tot_t[-20:])
            
            self.rew_1_t = np.append(self.rew_1_t, Rew_1)
            Rew_1 = np.mean(self.rew_1_t[-20:])
            
            self.rew_2_t = np.append(self.rew_2_t, Rew_2)
            Rew_2 = np.mean(self.rew_2_t[-20:])
            
            if hasattr(self, 'path'):
                np.savez(self.path,tot=self.tot_t, rew_1 = self.rew_1_t, rew_2 = self.rew_2_t)
            
            self.loss_t = np.append(self.loss_t, loss)
            loss = np.mean(self.loss_t[-20:])

            self.vis.line(
                X=torch.Tensor([eps]).cpu(),
                Y=torch.Tensor([tot_reward, Rew_1, Rew_2, loss]).unsqueeze(0).cpu(),
                win=self.value_window,
                update='append')





class SacAgent:

    def __init__(self, env, log_dir, state_mean, state_std,
                 num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6, prefer_num = 8, buf_num = 0,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0, cuda_device=0, q_frequency=1000, model_saved_step=100000, on_line=False):
        # yeh add
        self.on_line = on_line
        self.num_q = 2 #10
        self.gamma = gamma
        self.print_on = False

        self.env = env
        # self.state_mean = state_mean
        # self.state_std = state_std

        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False
        self.q_frequency = q_frequency
        self.model_saved_step = model_saved_step
        self.QM = QMonitor()
        
        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)
        print(self.env.observation_space.shape[0])

        # yeh: There is no reward_num attribute in the offline environment
        self.env.reward_num = self.env.obj_dim
        print(self.env.reward_num)

        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0]+self.env.reward_num,
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        # self.critic = TwinnedQNetwork(
        #     self.env.observation_space.shape[0],
        #     self.env.action_space.shape[0],
        #     self.env.reward_num,
        #     hidden_units=hidden_units).to(self.device)
        # self.critic_target = TwinnedQNetwork(
        #     self.env.observation_space.shape[0],
        #     self.env.action_space.shape[0],
        #     self.env.reward_num,
        #     hidden_units=hidden_units).to(self.device).eval()
        self.critic = MultiQNetwork(
            self.num_q,
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.reward_num,
            hidden_units=hidden_units).to(self.device)
        self.critic_target = MultiQNetwork(
            self.num_q,
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.reward_num,
            hidden_units=hidden_units).to(self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        # self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        # self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)
        self.qn_optim = [Adam(Q.parameters(), lr=lr) for Q in self.critic.Qn]

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MOPrioritizedMemory(
                memory_size, self.env.observation_space.shape, self.env.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:

            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MOMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.env.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step)

        #Q Replay Buffer
        self.Q_memory = QMemory(buf_num)
        self.cur_p = 0
        self.cur_e = 0
        self.qmem_p = 0
        self.qmem_e = 0

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        
        self.monitor = []
        self.tot_t = []
        self.reward_v = []
        if self.env.reward_num == 3:
            PREF_=np.load("3pref_table.npy")
        elif self.env.reward_num == 4:
            PREF_=np.load("4pref_table.npy")
        elif self.env.reward_num == 5:
            PREF_=np.load("5pref_table.npy")
        else:
            PREF_ = PREF
        for i in PREF_:
            self.tot_t.append([])
            self.reward_v.append([])

        
        self.set_num = prefer_num # set of ω'
        
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

        self.q1_loss = 0


    # def eval(self):
    #     self.policy.eval()
    #     self.critic.eval()
    #
    # def train(self):
    #     self.policy.train()
    #     self.critic.train()

    def load_dataset_to_memory(self, trajs):
        for traj in trajs:
            for step in range(traj['raw_rewards'].shape[0]):
                if self.per:
                    # We need to give true done signal with addition to masked done
                    # signal to calculate multi-step rewards.
                    self.memory.append(
                        traj['observations'][step],
                        traj['preference'][step],
                        traj['actions'][step],
                        traj['raw_rewards'][step],
                        traj['next_observations'][step],
                        int(traj["terminals"][step]),
                        1,
                        episode_done=int(traj["terminals"][step]))
                else:
                    # We need to give true done signal with addition to masked done
                    # signal to calculate multi-step rewards.

                    self.memory.append(
                        traj['observations'][step],
                        traj['preference'][step],
                        traj['actions'][step],
                        traj['raw_rewards'][step],
                        traj['next_observations'][step],
                        int(traj["terminals"][step]),
                        episode_done=int(traj["terminals"][step]))



    def get_pref(self):
        preference = np.random.rand( self.env.reward_num)
        preference = preference.astype(np.float32)
        preference /= preference.sum()
        return preference


    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def run_offline(self, trajs):
        returns = []

        mo_returns = []
        mo_rewards = []
        tot_rewards = []
        for traj in trajs:
            traj['rewards'] = np.sum(np.multiply(traj['raw_rewards'], traj['preference']), axis=1)
            tot_rewards.append(traj['raw_rewards'].sum())

            returns.append(traj['rewards'].sum())
            mo_returns.append(np.sum(traj['raw_rewards'], 0))
            mo_rewards.append(traj['raw_rewards'])

        # self.rewards_mean = np.concatenate(mo_rewards).mean(0)
        # self.rewards_std= np.concatenate(mo_rewards).std(0)
        # # self.rewards_max = np.concatenate(mo_rewards).max(0)
        # self.rewards_min = np.concatenate(mo_rewards).min(0)
        # self.rewards_max_min = np.concatenate(mo_rewards).max(0) - self.rewards_min

        self.plot = False
        if self.plot:
            mo_returns = np.stack(mo_returns, axis=-1)
            # plt.scatter(mo_returns[0], mo_returns[1], alpha=0.5, s=1)
            plt.scatter(range(len(mo_returns[0])), mo_returns[0], alpha=0.5, s=1)
            plt.show()

            plt.scatter(range(len(mo_returns[1])), mo_returns[1], alpha=0.5, s=1)
            plt.show()

            sorted_inds = np.argsort(tot_rewards)  # lowest to highest
            a = np.array(tot_rewards)
            plt.scatter(range(sorted_inds.shape[0]), a[sorted_inds], alpha=0.5, s=1)
            # plt.scatter(range(sorted_inds.shape[0]), a, alpha=0.5, s=1)
            plt.show()


        #
        # for i in range(sorted_inds.shape[0]):
        #     self.train_episode_offline(trajs[sorted_inds[i]])
        #     # if self.steps > self.num_steps:
        #     #     break

        # for traj in trajs:
        #     self.train_episode_offline(traj)
        #     if self.steps > self.num_steps:
        #         break

        for i in range(len(returns)):
            self.train_episode_offline(trajs[i])
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def act(self, state, preference=None):
        if preference is None:
            #rand = random.randint(0, len(PREF)-1)
            #preference = np.array(PREF[rand])
            preference = self.get_pref()
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state,preference)
        return action

    def explore(self, state, preference):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # self.eval()
            action, _, _ = self.policy.sample(state, preference)
            # self.train()
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state, preference):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # self.eval()
            _, _, action = self.policy.sample(state, preference)
            # self.train()
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, preference, actions, rewards, next_states, dones):
        curr_qn = self.critic(states, actions, preference)
        return curr_qn

    def calc_target_q(self, states, preference, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states, preference)
            next_qn = self.critic_target(next_states, next_actions, preference)
            
            #We choose argmin_Q (ωTQ)
            w_qn = [torch.einsum('ij,j->i', [next_q, preference[0]]) for next_q in next_qn]

            tmp_w_qn = torch.stack(w_qn, 1)
            indices = torch.min(tmp_w_qn, 1)[1]

            tmp_next_qn = torch.stack(next_qn, 1)
            inds = indices.unsqueeze(-1).repeat([1,self.env.reward_num]).unsqueeze(1)
            minq = torch.gather(tmp_next_qn, 1, inds).squeeze(1)
                
            next_q = minq + self.alpha * next_entropies     # yjyeh: q-alph.log(pi(next_action|next state)

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode_offline(self, traj):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        # Sample preference from prefernence space
        preference = self.get_pref()
        if self.env.reward_num == 3:
            PREF_ = np.load("3pref_table.npy")
        elif self.env.reward_num == 4:
            PREF_ = np.load("4pref_table.npy")
        elif self.env.reward_num == 5:
            PREF_ = np.load("5pref_table.npy")
        else:
            PREF_ = PREF

        for step in range(traj['raw_rewards'].shape[0]):
            ## Just fixed
            # action = self.act(state, preference)
            # # action = self.act(state)
            # next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += traj['raw_rewards'][step]

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            done = traj["terminals"][step]
            if episode_steps >= self.env.spec.max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    traj['observations'][step],
                    traj['preference'][step],
                    traj['actions'][step],
                    traj['raw_rewards'][step],
                    traj['next_observations'][step],
                    masked_done,
                    self.device)

                with torch.no_grad():
                    curr_qn = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_qn[0] - target_q).cpu().numpy()

                self.memory.append(
                    traj['observations'][step],
                    traj['preference'][step],
                    traj['actions'][step],
                    traj['raw_rewards'][step],
                    traj['next_observations'][step],
                    masked_done,
                    error,
                    episode_done=traj["terminals"][step])
            else:
                    # We need to give true done signal with addition to masked done
                    # signal to calculate multi-step rewards.
                # s = np.clip((traj['observations'][step] - self.state_mean) / self.state_std, -10, 10)
                # ns = np.clip((traj['next_observations'][step] - self.state_mean) / self.state_std, -10, 10)

                # step_rewards = (traj['raw_rewards'][step] - self.rewards_mean) / self.rewards_std
                # step_rewards = (traj['raw_rewards'][step] - self.rewards_min) / self.rewards_max_min
                self.memory.append(
                    traj['observations'][step],
                    # s,
                    traj['preference'][step],
                    traj['actions'][step],
                    traj['raw_rewards'][step],
                    # step_rewards,
                    traj['next_observations'][step],
                    # ns,
                    masked_done,
                    episode_done=int(traj["terminals"][step]))

                '''
                batch = to_batch(
                    state, preference, action, reward, next_state, masked_done,
                    self.device)

                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, preference, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.

                self.memory.append(
                    state, preference, action, reward, next_state, masked_done,
                    episode_done=done)
            '''

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.q1_loss = self.learn()

        # for ite in range(self.env.max_episode_steps): #self.n_steps_per_iter):
        #     for _ in range(self.updates_per_step):
        #         self.learn()

            if self.steps % self.eval_interval == 0:
                writer.add_scalar("train/loss", self.q1_loss, int(self.steps / self.eval_interval ))

                for i in range(len(PREF_)):
                    # self.evaluate(PREF_[i],self.monitor[i],i)
                    self.evaluate_(PREF_[i], i)

                if self.steps % self.model_saved_step == 0:
                    self.save_models(self.steps / self.model_saved_step)

        # preference = traj['preference'][0]
        # print(f'episode: {self.episodes:<4}  '
        #       f'episode steps: {episode_steps:<4}  '
        #       f'episode weight: {preference}  '
        #       f'reward:', episode_reward)


            if self.steps > self.num_steps:
                break

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        #Sample preference from prefernence space
        preference = self.get_pref()
        if self.env.reward_num == 3:
            PREF_=np.load("3pref_table.npy")
        elif self.env.reward_num == 4:
            PREF_=np.load("4pref_table.npy")
        elif self.env.reward_num == 5:
            PREF_=np.load("5pref_table.npy")
        else:
            PREF_ = PREF
        while not done:
            ## Just fixed
            action = self.act(state, preference)
            #action = self.act(state)
            # next_state, reward, done, _ = self.env.step(action)
            # episode_reward += reward
            next_state, _, done, info = self.env.step(action)
            reward = info['obj']
            episode_reward += reward

            self.steps += 1
            episode_steps += 1


            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env.spec.max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, preference, action, reward, next_state, masked_done,
                    self.device)
     
                with torch.no_grad():
                    curr_qn = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_qn[0] - target_q).cpu().numpy()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, preference, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.

                # s = np.clip((state - self.state_mean) / self.state_std, -10, 10)
                # ns = np.clip((next_state - self.state_mean) / self.state_std, -10, 10)
                # self.memory.append(
                #     s, preference, action, reward, ns, masked_done,
                #     episode_done=done)

                # step_rewards = (reward - self.rewards_mean) / self.rewards_std
                self.memory.append(
                    state, preference, action,
                    reward,
                    # step_rewards,
                    next_state, masked_done,
                    episode_done=done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.q1_loss = self.learn()

            if self.steps % self.eval_interval == 0:
                writer.add_scalar("train/loss", self.q1_loss, int(self.steps / self.eval_interval ))
                for i in range(len(PREF_)):
                    #self.evaluate(PREF_[i],self.monitor[i],i)
                    self.evaluate_(PREF_[i],i)
                if self.steps % self.model_saved_step == 0:
                    self.save_models(self.steps/self.model_saved_step)

            state = next_state

        # We log running mean of training rewards.
        # self.train_rewards.append(episode_reward)

        
        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'episode weight: {preference}  '
              f'reward:', episode_reward)

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.learning_steps % self.q_frequency == 0 and self.learning_steps > 20000:
            co = copy.deepcopy(self.critic)
            self.Q_memory.append(co)
        
        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.


        # Form preference set W containing the updating preference
        preference = self.get_pref()
        preference = torch.tensor(preference ,device = self.device)

        # q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
        qn_loss, errors, mean_qn = self.calc_critic_loss(batch, weights, preference)

        # if self.on_line:
        PREF_SET = []
        PREF_SET.append(preference)

        # indice = np.random.randint(low=0, high=self.batch_size, size = self.set_num-1)
        # a = [x for x in batch[1][indice]]
        # PREF_SET = PREF_SET + a

        for _ in range(self.set_num-1):
            p = self.get_pref()
            p = torch.tensor(p ,device = self.device)
            PREF_SET.append(p)
        #
        #     policy_loss, entropies = self.calc_policy_loss(batch, weights, preference, PREF_SET)
        # else:
        policy_loss, entropies = self.offline_calc_policy_loss(batch, weights, preference, PREF_SET, self.on_line)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip, print_on=self.print_on)


        # update_params(
        #     self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        # update_params(
        #     self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        for idx in range(len(self.qn_optim)):
            update_params(
                self.qn_optim[idx], self.critic.Qn[idx], qn_loss[idx], self.grad_clip, print_on=self.print_on)

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        # self.QM.update(self.steps, self.cur_p, self.cur_e, self.qmem_p, self.qmem_e)

        return qn_loss[0].item()

    def calc_critic_loss(self, batch, weights, preference):
        

        states, _, actions, rewards, next_states, dones = batch

        q1_losses = []
        q2_losses = []
        errorses = []
        mean_q1s = []
        mean_q2s = []

        
        D_pref = preference.repeat(self.batch_size,1)

        curr_qn = self.calc_current_q(states, D_pref, actions, rewards, next_states, dones)

        # writer.add_graph(self.critic, (states, actions, D_pref))

        target_q = self.calc_target_q(states, D_pref, actions, rewards, next_states, dones)

        # TD errors for updating priority weights
        # errors = torch.abs(curr_q1.detach() - target_q)
        # yeh: todo
        errors = torch.abs(curr_qn[0].detach() - target_q)
        # We log means of Q to monitor training.
        # mean_q1 = curr_q1.detach().mean().item()
        # mean_q2 = curr_q2.detach().mean().item()
        mean_qn = [curr_q.detach().mean().item() for curr_q in curr_qn]
      

        # Critic loss is mean squared TD errors with priority weights.
        # q1_loss = torch.mean(torch.tensordot((curr_q1 - target_q).pow(2), preference,dims=1) * weights)
        # q2_loss = torch.mean(torch.tensordot((curr_q2 - target_q).pow(2), preference,dims=1) * weights)
        qn_loss = [torch.mean(torch.tensordot((curr_q - target_q).pow(2), preference, dims=1) * weights) for curr_q in curr_qn]

        return qn_loss, errors, mean_qn

    # def offline_calc_policy_loss(self, batch, weights, preference, is_online):
    #     start = time.time()
    #     states, _, actions, rewards, next_states, mc_return, dones = batch
    #     preference_batch = preference.repeat(self.batch_size, 1)
    #
    #     # p_batch = torch.tensor(preference, device=self.device).repeat(self.batch_size, 1)
    #     # sampled_action, entropy, _ = self.policy.sample(states, p_batch)
    #
    #     sampled_action, entropy, _ = self.policy.sample(states, preference_batch)
    #     qn = self.critic(states, sampled_action, preference_batch)
    #
    #     # writer.add_graph(self.policy, (states, preference_batch))
    #
    #     w_qn = [torch.tensordot(q, preference, dims=1) for q in qn]
    #     q = torch.min(torch.stack(w_qn, 1), 1)[0]
    #
    #     # q1 = torch.tensordot(q1, preference, dims=1)
    #     # q2 = torch.tensordot(q2, preference, dims=1)
    #     # q = torch.min(q1, q2)f
    #     policy_loss = - q - self.alpha * (entropy.squeeze())
    #     policy_loss = torch.mean(policy_loss)
    #
    #     if not is_online:
    #         policy_loss = torch.abs(policy_loss.detach() * (1 - torch.mean(torch.nn.functional.cosine_similarity(actions, sampled_action)))
    #
    #     return policy_loss, entropy()

    def offline_calc_policy_loss(self, batch, weights, preference, PREF, is_online):
        start = time.time()
        states, _, actions, rewards, next_states, mc_return, dones = batch
        preference_batch = preference.repeat(self.batch_size * self.set_num, 1)

        prefs = torch.stack(PREF)
        prefs_batch = prefs.repeat(self.batch_size, 1)  # w1w2w3w4 w1w2w3w4...w1w2w3w4  4*256
        b_pref_states = states.unsqueeze(1).repeat(1, self.set_num, 1)  # dim = 256, 4, 17
        b_pref_states = b_pref_states.reshape(-1, b_pref_states.shape[-1])  # s1,s1,s1,s1,s2,s2,s2,s2,...dim = 256*4, 17

        sampled_action, entropy_prefs, _ = self.policy.sample(b_pref_states, preference_batch)

        entropy_prefs = entropy_prefs.squeeze()  # 256*4
        entropy_prefs = entropy_prefs.reshape(self.batch_size, -1)  # dim = [256, 4]
        entropy_batch = entropy_prefs[:, 0]  # dim = 256
        entropy = entropy_batch.unsqueeze(-1).repeat(1, 4).reshape(-1)  # dim = 256*4  (e1, e1, e1, e1, e2, e2, e2, e2, e3, ...)

        # sampled_action = sampled_action.squeeze()  # 256*4, 6
        tmp_sampled_action = sampled_action.reshape(self.batch_size, self.set_num, -1)  # dim = [256, 4]
        action_batch = tmp_sampled_action[:, 0, :]  # dim = 256

        losses = []

        c_cnt = 0
        # for a, c in enumerate([self.critic] + self.Q_memory.sample()):  # Use critic from Q Replay Buffer
        for a, c in enumerate([self.critic]):
            # if a == 0:
            #     entropy_prefs = entropy_prefs.squeeze()     # 256*4
            #     entropy_prefs = entropy_prefs.reshape(self.batch_size, -1)   # dim = [256, 4]
            #     entropy_batch = entropy_prefs[:, 0]   # dim = 256
            #     entropy = entropy_batch.unsqueeze(-1).repeat(1, 4).reshape(-1)  # dim = 256*4

            with torch.no_grad():
                qn = c(b_pref_states, sampled_action, prefs_batch)  # q(s1,w1) q(s1,w2), q(s1,w3), q(s1,w4), q(s2,w1),...

            w_qn = [torch.tensordot(q, preference, dims=1) for q in qn]
            q = torch.min(torch.stack(w_qn, 0), 0)[0]   # dim = 3 * 1024 = num_q * 1024  => min: 1024

            l = - q - self.alpha * entropy
            l = l.reshape(self.set_num, -1)     # dim = 4 * 246 = num_pref * batch
            losses.append(l)  # list[ 5 * [4, 256]]

        losses = torch.stack(losses, dim=0)  # dim = [5, 4, 256]
        losses = losses.reshape(-1, losses.shape[-1])    # dim = [20, 256]
        policy_loss, idx = torch.min(losses, 0)
        # ll=idx.detach().cpu()[:,0].tolist()
        policy_loss = torch.mean(policy_loss)

        # sampled_action, e, _ = self.policy.sample(states, preference_batch)
        if not is_online:
            # Aug12_06-26-12 good policy_loss = torch.abs(policy_loss.detach()) * (1 - torch.mean(torch.nn.functional.cosine_similarity(actions.double(), action_batch.double())))
            # Aug13_10-24 bad !! policy_loss = policy_loss - torch.abs(policy_loss.detach()) * torch.mean(torch.nn.functional.cosine_similarity(actions.double(), action_batch.double()))
            # policy_loss = policy_loss * torch.mean(torch.nn.functional.cosine_similarity(actions.double(), action_batch.double()))
            policy_loss = -1 * torch.abs(policy_loss.detach()) * torch.mean(torch.nn.functional.cosine_similarity(actions.double(), action_batch.double()))
        return policy_loss, entropy_batch


    # def calc_policy_loss_org(self, batch, weights, preference, PREF):
    #     start = time.time()
    #     states, _, actions, rewards, next_states, dones = batch
    #     preference_batch = preference.repeat(self.batch_size, 1)
    #
    #     losses = []
    #
    #     c_cnt = 0
    #     for a, c in enumerate([ self.critic]+self.Q_memory.sample() ): # Use critic from Q Replay Buffer
    #         for b, i in enumerate(PREF): #Get Q from preference set W
    #             p_batch = torch.tensor(i, device = self.device).repeat(self.batch_size, 1)
    #             sampled_action, entropy, _ = self.policy.sample(states, p_batch)
    #             if a == 0 and b == 0:
    #                 e = entropy
    #             qn = c(states, sampled_action, preference_batch)  #yeh ???
    #             w_qn = [torch.tensordot(q, preference, dims=1) for q in qn]
    #             q = torch.min(torch.stack(w_qn, 1), 1)[0]
    #
    #             # q1 = torch.tensordot(q1, preference, dims = 1)
    #             # q2 = torch.tensordot(q2, preference, dims = 1)
    #             # q = torch.min(q1, q2)
    #
    #             # l = - q - self.alpha * entropy
    #             l = - q - self.alpha * (entropy.squeeze())
    #             losses.append(l)
    #
    #     losses = torch.stack(losses, dim = 1)
    #     policy_loss, idx =  torch.min(losses, 1)
    #     # ll=idx.detach().cpu()[:,0].tolist()
    #     policy_loss = torch.mean(policy_loss)
    #
    #
    #     sampled_action, e, _ = self.policy.sample(states, preference_batch)
    #
    #     return policy_loss, e



    # def calc_policy_loss(self, batch, weights, preference, PREF):
    #     start = time.time()
    #     states, _, actions, rewards, next_states, dones = batch
    #     preference_batch = preference.repeat(self.batch_size * self.set_num, 1)
    #
    #     losses = []
    #
    #     c_cnt = 0
    #     for a, c in enumerate([self.critic] + self.Q_memory.sample()):  # Use critic from Q Replay Buffer
    #         prefs = torch.stack(PREF)
    #         prefs_batch = prefs.repeat(self.batch_size, 1)   #w1w2w3w4 w1w2w3w4...w1w2w3w4  4*256
    #         b_pref_states = states.unsqueeze(1).repeat(1, self.set_num, 1)   # dim = 256, 4, 17
    #         b_pref_states = b_pref_states.reshape(-1, b_pref_states.shape[-1])  # dim = 256*4, 17
    #
    #         sampled_action, entropy_prefs, _ = self.policy.sample(b_pref_states, preference_batch)
    #
    #         if a == 0:
    #             entropy_prefs = entropy_prefs.squeeze()     # 256*4
    #             entropy_prefs = entropy_prefs.reshape(self.batch_size, -1)   # dim = [256, 4]
    #             entropy_batch = entropy_prefs[:, 0]   # dim = 256
    #             entropy = entropy_batch.unsqueeze(-1).repeat(1, 4).reshape(-1)  # dim = 256*4
    #         qn = c(b_pref_states, sampled_action, prefs_batch)
    #         w_qn = [torch.tensordot(q, preference, dims=1) for q in qn]
    #         q = torch.min(torch.stack(w_qn, 1), 1)[0]
    #         l = - q - self.alpha * entropy
    #         l = l.reshape(self.set_num, -1)
    #         losses.append(l)  # list[ 5 * [4, 256]]
    #
    #     losses = torch.stack(losses, dim=0)  # dim = [5, 4, 256]
    #     losses = losses.reshape(-1, losses.shape[-1])    # dim = [20, 256]
    #     policy_loss, idx = torch.min(losses, 0)
    #     # ll=idx.detach().cpu()[:,0].tolist()
    #     policy_loss = torch.mean(policy_loss)
    #
    #     # yeh ??
    #     sampled_action, e, _ = self.policy.sample(states, preference_batch)
    #
    #     # return policy_loss, entropy_batch
    #     return policy_loss, e


    def calc_policy_loss(self, batch, weights, preference, PREF):
        start = time.time()
        states, _, actions, rewards, next_states, dones = batch
        preference_batch = preference.repeat(self.batch_size * self.set_num, 1)

        prefs = torch.stack(PREF)
        prefs_batch = prefs.repeat(self.batch_size, 1)  # w1w2w3w4 w1w2w3w4...w1w2w3w4  4*256
        b_pref_states = states.unsqueeze(1).repeat(1, self.set_num, 1)  # dim = 256, 4, 17
        b_pref_states = b_pref_states.reshape(-1, b_pref_states.shape[-1])  # s1,s1,s1,s1,s2,s2,s2,s2,...dim = 256*4, 17

        sampled_action, entropy_prefs, _ = self.policy.sample(b_pref_states, preference_batch)

        entropy_prefs = entropy_prefs.squeeze()  # 256*4
        entropy_prefs = entropy_prefs.reshape(self.batch_size, -1)  # dim = [256, 4]
        entropy_batch = entropy_prefs[:, 0]  # dim = 256
        entropy = entropy_batch.unsqueeze(-1).repeat(1, 4).reshape(-1)  # dim = 256*4  (e1, e1, e1, e1, e2, e2, e2, e2, e3, ...)

        losses = []

        c_cnt = 0
        # for a, c in enumerate([self.critic] + self.Q_memory.sample()):  # Use critic from Q Replay Buffer
        for a, c in enumerate([self.critic] + self.Q_memory.sample()):  # Use critic from Q Replay Buffer
            # if a == 0:
            #     entropy_prefs = entropy_prefs.squeeze()     # 256*4
            #     entropy_prefs = entropy_prefs.reshape(self.batch_size, -1)   # dim = [256, 4]
            #     entropy_batch = entropy_prefs[:, 0]   # dim = 256
            #     entropy = entropy_batch.unsqueeze(-1).repeat(1, 4).reshape(-1)  # dim = 256*4
            qn = c(b_pref_states, sampled_action, prefs_batch)  # q(s1,w1) q(s1,w2), q(s1,w3), q(s1,w4), q(s2,w1),...

            w_qn = [torch.tensordot(q, preference, dims=1) for q in qn]
            q = torch.min(torch.stack(w_qn, 0), 0)[0]   # dim = 3 * 1024 = num_q * 1024  => min: 1024

            l = - q - self.alpha * entropy
            l = l.reshape(self.set_num, -1)     # dim = 4 * 246 = num_pref * batch
            losses.append(l)  # list[ 5 * [4, 256]]

        losses = torch.stack(losses, dim=0)  # dim = [5, 4, 256]
        losses = losses.reshape(-1, losses.shape[-1])    # dim = [20, 256]
        policy_loss, idx = torch.min(losses, 0)
        # ll=idx.detach().cpu()[:,0].tolist()
        policy_loss = torch.mean(policy_loss)

        # sampled_action, e, _ = self.policy.sample(states, preference_batch)

        return policy_loss, entropy_batch

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def evaluate_(self, preference, ind):
        episodes = 5
        returns = np.empty((episodes,self.env.reward_num))
        preference = np.array(preference)
        for i in range(episodes):
            state = self.env.reset()
            episode_reward = np.zeros(self.env.reward_num)
            done = False
            trace = []
            actions = []
            while not done:
                # s = np.clip((state - self.state_mean) / self.state_std, -10, 10)
                # action = self.exploit(s, preference)
                action = self.exploit(state, preference)
                trace.append(list(state))
                actions.append(list(action))
                # next_state, reward, done, _ = self.env.step(action)
                # episode_reward += reward
                next_state, _, done, info = self.env.step(action)
                episode_reward += info['obj']
                state = next_state

            returns[i] = episode_reward
            print(episode_reward)
        mean_return = np.mean(returns, axis=0)


        # batch = self.memory.sample(self.batch_size)
        # p = torch.tensor(preference ,device = self.device, dtype=torch.float32)
        # with torch.no_grad():
        #     q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
        #                     self.calc_critic_loss(batch, 1, p, 0)
        # # monitor.update(self.steps/self.eval_interval, np.dot(preference,mean_return), *mean_return, q1_loss.mean().item())

        eval_ep = int(self.steps / self.eval_interval)
        # writer.add_scalar("train/loss", q1_loss.mean().item(), eval_ep)
        writer.add_scalar("eval/tot_return" + np.array2string(preference, formatter={'float_kind': lambda x: "%.2f" % x}), np.dot(preference,mean_return), eval_ep)

        pref_str = np.array2string(preference, formatter={'float_kind': lambda x: "%.2f" % x}) + "/obj"
        for i in range(mean_return.shape[0]):
            writer.add_scalar("eval/" + pref_str + str(i), mean_return[i], eval_ep)


        path = os.path.join(self.log_dir, 'summary')
        tot_path = os.path.join(path, f'{ind}total_log.npy')
        reward_path = os.path.join(path, f'{ind}reward_log.npy')
        self.tot_t[ind].append( np.dot(preference, mean_return) )
        self.reward_v[ind].append(mean_return)

        np.save(tot_path, np.array(self.tot_t[ind]) )
        np.save(reward_path, np.array(self.reward_v[ind]) )

        print('-' * 60)
        print(f'preference ', preference,
              f'Num steps: {self.steps:<5}  '
              f'reward:', mean_return)
        print('-' * 60)

    def evaluate(self, preference, monitor, ind):
        episodes = 10
        returns = np.empty((episodes,self.env.reward_num))
        preference = np.array(preference)
        for i in range(episodes):
            state = self.env.reset()
            episode_reward = np.zeros(self.env.reward_num)
            done = False
            trace = []
            actions = []
            while not done:
                # s = np.clip((state - self.state_mean) / self.state_std, -10, 10)
                # action = self.exploit(s,preference )
                action = self.exploit(state, preference)
                trace.append(list(state))
                actions.append(list(action))
                # next_state, reward, done, _ = self.env.step(action)
                # episode_reward += reward
                next_state, _, done, info = self.env.step(action)
                episode_reward += info['obj']
                state = next_state

            returns[i] = episode_reward
            print(episode_reward)
        mean_return = np.mean(returns, axis=0)
        
        batch = self.memory.sample(self.batch_size) 
        p = torch.tensor(preference ,device = self.device, dtype=torch.float32)
        with torch.no_grad():
            # q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            qn_loss, errors, mean_q = self.calc_critic_loss(batch, 1, p, 0)
        #monitor.update(self.steps/self.eval_interval, np.dot(preference,mean_return), *mean_return, q1_loss.mean().item())
        #monitor.update(self.steps / self.eval_interval, np.dot(preference, mean_return), *mean_return, qn_loss[0].mean().item())


        path = os.path.join(self.log_dir, 'summary')
        tot_path = os.path.join(path, f'{p_name[ind]}total_log.npy')
        reward_path = os.path.join(path, f'{p_name[ind]}reward_log.npy')
        self.tot_t[ind].append( np.dot(preference, mean_return) )
        self.reward_v[ind].append(mean_return)

        np.save(tot_path, np.array(self.tot_t[ind]) )
        np.save(reward_path, np.array(self.reward_v[ind]) )

        print('-' * 60)
        print(f'preference ', preference,
              f'Num steps: {self.steps:<5}  '
              f'reward:', mean_return)
        print('-' * 60)

    def save_models(self, num):
        self.policy.save(os.path.join(self.model_dir, 'policy_'+str(num)+'.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic_'+str(num)+'.pth'))
        self.critic_target.save(os.path.join(self.model_dir, 'critic_target'+str(num)+'.pth'))
        with open(os.path.join(self.model_dir, 'Q_memory_'+str(num)+'.pkl'), 'wb') as file:
            pickle.dump(self.Q_memory, file)
        with open(os.path.join(self.model_dir, 'replay_buf_'+str(num)+'.pkl'), 'wb') as file:
            pickle.dump(self.memory, file)

    def load_models(self, num):
        self.policy.load(os.path.join(self.model_dir, 'policy_'+str(num)+'.pth'))
        self.critic.load(os.path.join(self.model_dir, 'critic_'+str(num)+'.pth'))
        self.critic_target.load(os.path.join(self.model_dir, 'critic_target'+str(num)+'.pth'))
        with open(os.path.join(self.model_dir, 'Q_memory_'+str(num)+'.pkl'), 'rb') as file:
            self.Q_memory = pickle.load(file)
        with open(os.path.join(self.model_dir, 'replay_buf_'+str(num)+'.pkl'), 'rb') as file:
            self.memory = pickle.load(file)

    def __del__(self):
        #self.writer.close()
        self.env.close()
