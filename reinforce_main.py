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

agentParam = {"gamma":args.gamma,"LR":1e-2,"device":device}

model_name = "pg_social"
file_name  = "train_para/"+model_name
save_eps = 10
ifsave_model = True
class independentAgent():
    def __init__(self,agents):
        self.n_agents = len(agents)
        ## agents : list of agents
        self.agents = agents

    def select_actions(self,state):
        ## state: [obs1,obs2,...]
        actions = []
        for i,ag in zip(range(self.n_agents),self.agents):
            action = ag.select_action(state[i])
            actions.append(action)
        return actions
    
    def push_reward(self,reward):
        for i,ag in zip(range(self.n_agents),self.agents):
            ag.rewards.append(reward[i])
            
    def update_agents(self):
        for ag in self.agents:
            ag.update()
    
    def save(self,file_name):
        for i,ag in zip(range(self.n_agents),self.agents):
            torch.save(ag,file_name+"pg"+str(i)+".pth") 


class socialAgents(independentAgent):
    def __init__(self,agents,agentParam):
        super().__init__(agents)
        self.Law = social_agent(agentParam)

    def select_mask_actions(self,state):
        actions = []
        for i,ag in zip(range(self.n_agents),self.agents):
            masks, prob_mask= self.Law.select_action(state[i])
            self.Law.prob_social.append(prob_mask)
            pron_mask_copy = prob_mask#deepcopy(prob_mask)
            action, prob_indi = ag.select_mask_action(state[i],pron_mask_copy)
            self.Law.pi_step.append(prob_indi)
            actions.append(action)
        return actions
    def push_reward(self,reward):
        for i,ag in zip(range(self.n_agents),self.agents):
            ag.rewards.append(reward[i])
        self.Law.rewards.append(sum(reward))
    def update_law(self):
        self.Law.update(self.n_agents)
    def save(self,file_name):
        torch.save(self.Law,file_name+"pg_law"+".pth")
        for i,ag in zip(range(self.n_agents),self.agents):
            torch.save(ag,file_name+"pg"+str(i)+".pth") 
def main():
    #agent = PGagent(agentParam)
    n_agents = 2
    #multiPG = independentAgent([PGagent(agentParam) for i in range(n_agents)])
    multiPG = socialAgents([newPG(agentParam) for i in range(n_agents)],agentParam)
    for i_episode in range(101):
        n_state, ep_reward = env.reset(), 0
        for t in range(1, 500):
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

        if i_episode % save_eps == 0 and i_episode>1 and ifsave_model:
            multiPG.save(file_name)
if __name__ == '__main__':
    main()