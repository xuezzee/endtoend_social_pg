import argparse
import gym
import numpy as np
from itertools import count
import random
import torch
import time
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from gather_env import GatheringEnv
from endtoend_social_pg.envs.SocialDilemmaENV.social_dilemmas.envir.Coin_game import CoinGameVec
from envtest import envSocialDilemma
from PGagent import PGagent,social_agent,newPG,IAC,Centralised_AC
from network import socialMask,Centralised_Critic
from copy import deepcopy
import os
from logger import Logger
from multiAG import independentAgent,socialAgents,AC_Agents
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

n_agents = 2
env = CoinGameVec(1000,1,5)
# env = envSocialDilemma("cleanup",n_agents)
#env.seed(args.seed)
torch.manual_seed(args.seed)

test_mode = True
env_dim =  {"harvest":[363,8],"cleanup":[16,8]}
names = {"social":"pg_social","base":"pg_indi_lessApple_","single":"pg_single_less","forbid":"base_forbid","base_harv":"small_harvest"
,"harv_single": "harv_single","AC":"AC_harv","cleanup":"cleanup_mini"}
mode = "AC"
model_name = names[mode]
model_name = "small_harvest"
file_name  = "train_para/"+model_name
agentParam = {"gamma":args.gamma,"LR":0.005,"device":device,"ifload":False,"filename": file_name}
save_eps = 100
impath = "images/harvest_small"
# impath = None

n_episode = 201
n_steps = 500
logger = Logger('./logs')

if test_mode:
    ifsave_model = False
    ifsave_data = False
    render = True
else:
    ifsave_model = True
    ifsave_data = True
    render = False


def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def random_agent():
    all_rw = []
    for i_episode in range(n_episode):
        ep_rw = 0
        for t in range(n_steps):
            actions = [ random.randint(0,7) for i in range(n_agents)]
            n_state, n_reward, _, _ = env.step(actions)
            ep_rw+=sum(n_reward)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_rw, ep_rw))
        all_rw.append(ep_rw)
        np.save("data/"+"random1"+".npy",all_rw)

class Agents():
    def __init__(self,agents,state_dim):
        self.num_agent = len(agents)
        self.agents = agents
        self.critic = Centralised_Critic(state_dim)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(),lr=0.01)
        self.lr_schedulerC = torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=1000, gamma=1, last_epoch=-1)
        for i in self.agents:
            i.critic = self.critic

    def choose_action(self,state):
        actions = []
        for agent, s in zip(self.agents, state):
            actions.append(agent.choose_action(s).detach().numpy())
        return actions

    def td_err(self, s, r, s_):
        s = torch.Tensor(s).reshape((1,-1)).unsqueeze(0)
        s_ = torch.Tensor(s_).reshape((1,-1)).unsqueeze(0)
        v = self.critic(s)
        v_ = self.critic(s_).detach()
        return r + 0.9*v_ - v

    def LearnCenCritic(self, s, r, s_):
        td_err = self.td_err(s,r,s_)
        # m = torch.log(self.agents.act_prob[a[0]]*self.agents.act_prob[a[1]])
        loss = torch.square(td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_schedulerC.step()

    def update(self, state, reward, state_, action):
        td_err = self.td_err(state[0],sum(reward),state_[0])
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a[0],td_err[0])
        self.LearnCenCritic(state[0],sum(reward),state_[0])


def main():
    #agent = PGagent(agentParam)
    all_rw = []
    #n_agents = 1#2
    if mode == "social":
        multiPG = socialAgents([PGagent(env_dim["cleanup"][0],env_dim["cleanup"][1],add_para(i)) for i in range(n_agents)],agentParam)
    else:
        multiPG = Agents([PGagent(env_dim["cleanup"][0],env_dim["cleanup"][1],add_para(i)) for i in range(n_agents)])

    for i_episode in range(n_episode):
        n_state, ep_reward = env.reset_linear(), 0
        for t in range(n_steps):
            if mode == "social":
                actions = multiPG.select_mask_actions(n_state)
            else:
                actions = multiPG.select_actions(n_state)##agent.select_action(state)
            #actions = [ random.randint(0,7) for i in range(n_agents)]
            n_state, n_reward, _, _ = env.step_linear(actions)
            if render and i_episode==0:
                env.render(impath,t)
            multiPG.push_reward(n_reward)
            ep_reward += sum(n_reward)
            multiPG.update_agents()

        running_reward = ep_reward
        # if test_mode == False:

        all_rw.append(ep_reward)
        if i_episode % (args.log_interval*2) == 0 and ifsave_data:
            np.save("data/"+model_name+".npy",all_rw)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if i_episode % save_eps == 0 and i_episode>10 and ifsave_model:
            multiPG.save(file_name)

def process_state(state):
    s = deepcopy(state)
    for i in range(len(state)):
        s[i] = state[i][0]
    return s

def AC_main():
    #agent = PGagent(agentParam)
    all_rw = []
    #n_agents = 1#2
    if mode == "social":
        multiPG = socialAgents([PGagent(env_dim["cleanup"][0],env_dim["cleanup"][1],add_para(i)) for i in range(n_agents)],agentParam)
    elif mode == "AC":
        multiPG = Agents([Centralised_AC(4,100) for i in range(n_agents)],50 )
    else:
        multiPG = independentAgent([PGagent(env_dim["cleanup"][0],env_dim["cleanup"][1],add_para(i)) for i in range(n_agents)])

    for i_episode in range(1000):
        n_state, ep_reward = env.reset(), 0
        n_state = n_state[0]
        test_reward_sum = 0
        for t in range(1000):

            if mode == "social":
                actions = multiPG.select_mask_actions(n_state)
            else:
                actions = multiPG.choose_action(process_state(n_state))##agent.select_action(state)
            #actions = [ random.randint(0,7) for i in range(n_agents)]
            a = deepcopy(actions)
            for i in range(len(actions)):
                a[i] = [actions[i][0]]
            n_state_, n_reward, _, _, test_reward = env.step(a)
            test_reward_sum+=test_reward
            if render and i_episode!=1:
                # env.render(impath,t)
                env.render()
            # time.sleep(0.05)
            #multiPG.push_reward(n_reward)
            ep_reward += sum(n_reward)
            # if [1] in process_state(n_reward):
            #     print("i_episode %d:"%i_episode,process_state(n_reward))
            multiPG.update(process_state(n_state), process_state(n_reward), process_state(n_state_), actions)
            n_state = n_state_
        running_reward = ep_reward
        #if test_mode == False:
        #    multiPG.update_agents()

        all_rw.append(ep_reward)
        if i_episode % (args.log_interval*2) == 0 and ifsave_data:
            np.save("data/"+model_name+".npy",all_rw)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\ttest_reward: {:.2f}'.format(
                  i_episode, ep_reward[0], running_reward[0], test_reward_sum))
            logger.scalar_summary("ep_reward", ep_reward, i_episode)
            logger.scalar_summary("coin_eaten",test_reward_sum,i_episode)

        if i_episode % save_eps == 0 and i_episode>10 and ifsave_model:
            multiPG.save(file_name)
if __name__ == '__main__':
    AC_main()
    #random_agent()