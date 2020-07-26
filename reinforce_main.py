import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gather_env import GatheringEnv
from PGagent import PGagent,social_agent,newPG
from network import socialMask
from copy import deepcopy
from multiAG import independentAgent,socialAgents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = GatheringEnv(2)#gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


model_name = "pg_social"
file_name  = "train_para/"+model_name
agentParam = {"gamma":args.gamma,"LR":1e-2,"device":device,"ifload":True,"filename": file_name}
save_eps = 10
ifsave_model = True
n_episode = 4#101
n_steps = 500

def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def main():
    #agent = PGagent(agentParam)
    n_agents = 2
    #multiPG = independentAgent([PGagent(agentParam) for i in range(n_agents)])
    multiPG = socialAgents([newPG(add_para(i)) for i in range(n_agents)],agentParam)
    for i_episode in range(n_episode):
        n_state, ep_reward = env.reset(), 0
        for t in range(n_steps):
            actions = multiPG.select_mask_actions(n_state)#agent.select_action(state)
            n_state, n_reward, _, _ = env.step(actions)
            if args.render:
                env.render()
            multiPG.push_reward(n_reward)
            ep_reward += sum(n_reward)

        running_reward = ep_reward
        multiPG.update_agents()
        multiPG.update_law()
        
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        if i_episode % save_eps == 0 and i_episode>10 and ifsave_model:
            multiPG.save(file_name)


if __name__ == '__main__':
    main()