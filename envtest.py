from envs.SocialDilemmaENV.social_dilemmas.envir.harvest import HarvestEnv
import random

from gather_env import GatheringEnv

class envGather:
    def __init__(self,n_agents,map_name='default_small'):
        self.n_agents = n_agents
        self.env = GatheringEnv(n_agents)
    
    def step(self,actions):
        return

class envSocialDilemma:
    def __init__(self):
        self.n_agents = 2
        self.world =  HarvestEnv(num_agents=self.n_agents)

    def step_linear(self,actions):
        ## input: [1,2,4] ...
        actions = self.transferData(actions,'a2')
        state,reward,done,info = self.world.step(actions)
        reward = self.transferData(reward,'r')
        state = self.transferData(state,'sc')
        return state,reward,done,info

    def step(self,actions):
        ## input: [1,2,4] ...
        ## to CNN 
        actions = self.transferData(actions,'a2')
        state,reward,done,info = self.world.step(actions)
        reward = self.transferData(reward,'r')
        state = self.transferData(state,'sc')
        state_2d = [ sa.T/255 for sa in state]
        return state_2d,reward,done,info

    def reset_linear(self):
        state = self.world.reset()
        state = self.transferData(state,'sc')
        return state

    def reset(self):
        state = self.world.reset()
        state = self.transferData(state,'sc')
        state_2d = [ sa.T/255 for sa in state]
        return state_2d
    
    def transferData(self,data,mode):
        def actionDict(actions):
            ## actions: list of action for each agent
            ## return dict of agents' action e.g. {agent-1:2,...}
            action_dict = {}
            for i in range(len(actions)):
                name = "agent-"+str(i)
                action_dict[name] = actions[i].item()
            return action_dict
        def transReward(reward_dict):
            ## sutible for reward and obs in AC
            rewards = []
            for key,value in reward_dict.items():
                rewards.append(value)
            return rewards
        def actionDict2(actions):
            ## actions: list of action for each agent
            ## return dict of agents' action e.g. {agent-1:2,...}
            action_dict = {}
            for i in range(len(actions)):
                name = "agent-"+str(i)
                action_dict[name] = actions[i]#.item()
            return action_dict
        def transPicture(pic_data):
            pics = []
            for key,value in pic_data.items():
                pics.append(value.T/255)
            #value0 = [sa.T/255 for sa in sta]
            return pics
        if mode == 'r':
            return transReward(data)
        elif mode == 'a':
            return actionDict(data)
        elif mode == 'ddpg_s':
            return transPicture(data)
        elif mode == 'sc':
            return transReward(data)
        elif mode == "a2":
            return actionDict2(data)
        else:
            return []

if __name__ == "__main__":
    world = envSocialDilemma()
    actions = [ random.randint(0,7) for i in range(world.n_agents)]
    #actions = world.transferData(actions,'a2')
    state,reward,_,_ = world.step(actions)
    print(state[0].shape)
    print(reward)