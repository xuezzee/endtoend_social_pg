import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from network import Policy,socialMask,Actor,Critic,CNN_preprocess,CriticRNN,ActorRNN
import copy
import itertools
import random

class PGagent():
    def __init__(self,agentParam):
        self.state_dim = 400#env.observation_space.shape[0]
        self.action_dim = 8#env.action_space.n
        self.gamma = agentParam["gamma"]
        # init N Monte Carlo transitions in one game
        self.saved_log_probs = []
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.rewards = []
        self.device = agentParam["device"]
        # init network parameters
        self.policy = Policy(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=agentParam["LR"])
        self.eps = np.finfo(np.float32).eps.item()

        # init some parameters
        self.time_step = 0


    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item()


    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device).type(self.FloatTensor)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss)
        policy_loss = policy_loss.sum()
        temp = copy.copy(policy_loss)
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return temp


class social_agent(PGagent):
    def __init__(self,agentParam):
        super().__init__(agentParam)
        self.policy = socialMask(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        #    #[[pi_1,pi_2,...],[pi_1,pi_2,...],...
        self.pi_step = []
        #    #[[prob_1,prob_2,...],[prob_1,prob_2,...],....
        self.prob_social = []
        # self.reward = [sum(R1),sum(R2),....]
    def select_action(self,state):                      #select
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        #self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),probs

    def maskFunc(self,probs,masks):
        return F.softmax(torch.mul(probs,masks))
    def update(self,n_agents):
        R = 0
        policy_loss = []
        returns_sum = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns_sum.insert(0, R)
        returns = [ [r]*n_agents for r in returns_sum]
        self.saved_log_probs = []
        for k in range(len(self.pi_step)):                       #pi_step is the list of unmasked policy(prob ditribution) for each agent
            pi = self.pi_step[k].detach()
            new_probs = self.maskFunc(pi,self.prob_social[k])    #prob_social is the list of masks for each agent
            m = Categorical(new_probs)
            action = m.sample()                                  #sample from the distribution
            self.saved_log_probs.append(m.log_prob(action).to(self.device)) #save the logged prob for sampled action
        returns = np.array(returns).flatten()
        returns = torch.tensor(returns).to(self.device).type(self.FloatTensor)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):   #calculate the -log(pi)R
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()                                   #back propagation
        self.optimizer.step()
        del self.rewards[:]
        del self.pi_step[:]
        del self.prob_social[:]


class newPG(PGagent):
    def __init__(self,agentParam):
        super().__init__(agentParam)
    def maskFunc(self,probs,masks):
        temp = torch.mul(probs,masks)
        # temp[0,0] = 1
        # for i in range(1,8):
        #     temp[0,i] = 0
        return F.softmax(temp)
    def select_masked_action(self,state,masks):
        masks = masks.detach()
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        new_probs = self.maskFunc(probs,masks)
        m = Categorical(new_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),probs

# class Actor(nn.Module):
#     def __init__(self,action_dim,state_dim):
#         super(Actor,self).__init__()
#         self.Linear1 = nn.Linear(state_dim,128)
#         # self.Dropout1 = nn.Dropout(p=0.3)
#         self.Linear2 = nn.Linear(128,action_dim)
#
#     def forward(self,x):
#         x = self.Linear1(x)
#         # x = self.Dropout1(x)
#         x = F.relu(x)
#         x = self.Linear2(x)
#         return F.softmax(x)
#
# class Critic(nn.Module):
#     def __init__(self,state_dim):
#         super(Critic,self).__init__()
#         self.Linear1 = nn.Linear(state_dim, 128)
#         # self.Dropout1 = nn.Dropout(p=0.3)
#         self.Linear2 = nn.Linear(128, 1)
#
#     def forward(self,x):
#         x = self.Linear1(x)
#         # x = self.Dropout1(x)
#         x = F.relu(x)
#         x = self.Linear2(x)
#         return x

class IAC():
    def __init__(self,action_dim,state_dim,CNN=False, width=None, height=None, channel=None):
        self.CNN = CNN
        if CNN:
            self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            self.CNN_preprocessC = CNN_preprocess(width,height,channel)
            state_dim = self.CNN_preprocessA.get_state_dim()
        self.actor = Actor(action_dim,state_dim)
        self.critic = Critic(state_dim)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.noise_epsilon = 0.999
        self.constant_decay = 1
        self.optimizerA = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(), lr = 0.001)
        self.lr_scheduler = {"optA":torch.optim.lr_scheduler.StepLR(self.optimizerA,step_size=1000,gamma=0.9,last_epoch=-1),
                             "optC":torch.optim.lr_scheduler.StepLR(self.optimizerC,step_size=1000,gamma=0.9,last_epoch=-1)}
        if CNN:
            # self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            # self.CNN_preprocessC = CNN_preprocess
            self.optimizerA = torch.optim.Adam(itertools.chain(self.CNN_preprocessA.parameters(),self.actor.parameters()),lr=0.0001)
            self.optimizerC = torch.optim.Adam(itertools.chain(self.CNN_preprocessC.parameters(),self.critic.parameters()),lr=0.001)
            self.lr_scheduler = {"optA": torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=10000, gamma=0.9, last_epoch=-1),
                                 "optC": torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=10000, gamma=0.9, last_epoch=-1)}
        # self.act_prob
        # self.act_log_prob

    def choose_action(self,s):
        s = torch.Tensor(s).unsqueeze(0)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1,3,15,15)))
        self.act_prob = self.actor(s) + torch.abs(torch.randn(self.action_dim)*0.*self.constant_decay)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def cal_tderr(self,s,r,s_):
        s = torch.Tensor(s).unsqueeze(0)
        s_ = torch.Tensor(s_).unsqueeze(0)
        if self.CNN:
            s = self.CNN_preprocessC(s.reshape(1,3,15,15))
            s_ = self.CNN_preprocessC(s_.reshape(1,3,15,15))
        v_ = self.critic(s_).detach()
        v = self.critic(s)
        return r + 0.9*v_ - v

    def learnCritic(self,s,r,s_):
        td_err = self.cal_tderr(s,r,s_)
        loss = torch.square(td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()

    def learnActor(self,s,r,s_,a):
        td_err = self.cal_tderr(s,r,s_)
        m = torch.log(self.act_prob[0][a]) #in cleanup there should not be a [0], in IAC the [0] is necessary
        temp = m*td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a):
        self.learnCritic(s,r,s_)
        self.learnActor(s,r,s_,a)

class Centralised_AC(IAC):
    def __init__(self,action_dim,state_dim):
        super().__init__(action_dim,state_dim)
        self.critic = None

    # def cal_tderr(self,s,r,s_):
    #     s = torch.Tensor(s).unsqueeze(0)
    #     s_ = torch.Tensor(s_).unsqueeze(0)
    #     v = self.critic(s).detach()
    #     v_ = self.critic(s_).detach()
    #     return r + v_ - v

    def learnActor(self,a,td_err):
        m = torch.log(self.act_prob[a])
        temp = m*td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a,td_err):
        self.learnActor(a,td_err)

class social_IAC(IAC):
    def __init__(self,action_dim,state_dim,agentParam):
        super().__init__(action_dim,state_dim)
        self.saved_log_probs = []
        self.device = agentParam["device"]
        self.reward = []
        self.rewards = []

    def select_masked_action(self,state,masks):
        masks = masks.detach()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.act_prob = self.actor(state)[0]
        new_probs = self.maskFunc(self.act_prob.detach(),masks)
        m = Categorical(new_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),self.act_prob

    def maskFunc(self,prob,mask):
        return F.softmax(torch.mul(prob,mask))

'''
This is the RNN version of Actor Crtic, in order to address the kind of problems where the temporal features are included
'''

class IAC_RNN(IAC):
    def __init__(self,action_dim,state_dim,CNN=True):
        super().__init__(action_dim,state_dim)
        self.maxsize_queue = 100
        self.CNN = CNN
        if CNN:
            self.queue = deque([torch.zeros(15**2*3).reshape(1,15,15,3) for i in range(self.maxsize_queue)])
        else:
            self.queue = deque([torch.zeros(state_dim).reshape(1,state_dim) for i in range(self.maxsize_queue)])
        self.actor = ActorRNN(state_dim,action_dim,CNN)
        self.critic = CriticRNN(state_dim,action_dim,CNN)
        self.optimizerA = torch.optim.Adam(self.actor.parameters(),lr=0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(),lr=0.01)


    def input_preprocess(self):
        return torch.Tensor

    def collect_states(self,state):
        self.queue.pop()
        self.queue.insert(0,state)

    def choose_action(self,s):
        s = torch.Tensor(s).unsqueeze(0)
        self.collect_states(s)
        self.queue.reverse()
        if self.CNN:
            self.act_prob = self.actor(torch.cat(list(self.queue)).reshape((-1,3,15,15))) + torch.abs(
                torch.randn(self.action_dim) * 0. * self.constant_decay)
        else:
            # t = torch.cat(list(self.queue)).reshape((-1,self.state_dim))
            self.act_prob = self.actor(torch.cat(list(self.queue)).reshape((1,-1,self.state_dim))) + torch.abs(
                torch.randn(self.action_dim) * 0. * self.constant_decay)
        self.queue.reverse()
        # self.act_prob = self.actor(s) + torch.abs(torch.randn(self.action_dim)*0.05*self.constant_decay)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        # self.act_prob[0][4]=0.
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def cal_tderr(self,s,r,s_):
        s = torch.Tensor(s).unsqueeze(0)
        s_ = torch.Tensor(s_).unsqueeze(0)
        temp_q = copy.deepcopy(self.queue)
        temp_q.pop()
        temp_q.insert(0,s_)
        temp_q.reverse()
        if self.CNN:
            v_ = self.critic(torch.cat(list(temp_q)).reshape((-1,3,15,15))).detach()
            self.queue.reverse()
            v = self.critic(torch.cat(list(self.queue)).reshape(-1,3,15,15))
            self.queue.reverse()
        else:
            v_ = self.critic(torch.cat(list(temp_q)).reshape((1,-1,self.state_dim))).detach()
            self.queue.reverse()
            v = self.critic(torch.cat(list(self.queue)).reshape(1,-1,self.state_dim))
            self.queue.reverse()
        return r + 0.97*v_ - v