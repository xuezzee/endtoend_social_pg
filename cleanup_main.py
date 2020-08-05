import argparse
import numpy as np
from gym.spaces import Discrete, Box
import torch
from cleanup import CleanupEnv
from PGagent import social_agent, IAC_RNN
from endtoend_social_pg.MAAC.algorithms.attention_sac import AttentionSAC
from endtoend_social_pg.MAAC.utils.buffer import ReplayBuffer
from endtoend_social_pg.parallel_env_process import envs_dealer
from logger import Logger

# from envs.ElevatorENV import Lift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# env = GatheringEnv(2)  # gym.make('CartPole-v1')
# env.seed(args.seed)
# torch.manual_seed(args.seed)

agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device}
# agentParam =

model_name = "pg_social"
file_name = "/Users/xue/Desktop/Social_Law/saved_weight/" + model_name
save_eps = 10
ifsave_model = True
logger = Logger('./logs')

class env_wrapper():
    def __init__(self,env):
        self.env = env

    def step(self,actions):
        def action_convert(action):
            # action = list(action.values())
            act = {}
            for i in range(len(action)):
                act["agent-%d"%i] = np.argmax(action[i],0)
            return act
        n_state_, n_reward, done, info = self.env.step(action_convert(actions))
        n_state_ = np.array([state.reshape(-1) for state in n_state_.values()])
        n_reward = np.array([reward for reward in n_reward.values()])
        return n_state_/255., n_reward, done, info

    def reset(self):
        n_state = self.env.reset()
        return np.array([state.reshape(-1) for state in n_state.values()])/255.

    def seed(self,seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()

    @property
    def observation_space(self):
        return Box(0., 1., shape=(675,), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def num_agents(self):
        return self.env.num_agents


def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = env_wrapper(CleanupEnv(num_agents=4))
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env()
    # if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    return envs_dealer([get_env_fn(i) for i in range(n_rollout_threads)])




class Agents():
    def __init__(self,agents,exploration=0.5):
        self.num_agent = len(agents)
        self.agents = agents
        self.exploration = exploration
        self.epsilon = 0.95


    def choose_action(self,state):
        actions = {}
        agentID = list(state.keys())
        i = 0
        for agent, s in zip(self.agents, state.values()):
            actions[agentID[i]] = int(agent.choose_action(s/255.).detach().numpy())
            i += 1
        return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, list(state), list(reward), list(state_), list(action)):
            agent.update(s/255.,r,s_/255.,a)

    def save(self, file_name):
        for i, ag in zip(range(self.num_agent), self.agents):
            torch.save(ag.policy, file_name + "pg" + str(i) + ".pth")

class Social_Agents():
    def __init__(self,agents,agentParam):
        self.Law = social_agent(agentParam)
        self.agents = agents
        self.n_agents = len(agents)

    def select_masked_actions(self, state):
        actions = []
        for i, ag in zip(range(self.n_agents), self.agents):
            masks, prob_mask = self.Law.select_action(state[i])
            self.Law.prob_social.append(prob_mask)  # prob_social is the list of masks for each agent
            pron_mask_copy = prob_mask  # deepcopy(prob_mask)
            action, prob_indi = ag.select_masked_action(state[i], pron_mask_copy)
            self.Law.pi_step.append(prob_indi)  # pi_step is the list of unmasked policy(prob ditribution) for each agent
            actions.append(action)
        return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a)

    def update_law(self):
        self.Law.update(self.n_agents)

    def push_reward(self, reward):
        for i, ag in zip(range(self.n_agents), self.agents):
            ag.rewards.append(reward[i])
        self.Law.rewards.append(sum(reward))


def main():
    env = CleanupEnv(num_agents=2)
    # agent = PGagent(agentParam)
    # writers = [writer = SummaryWriter('runs/fashion_mnist_experiment_1')]
    n_agents = 4
    # multiPG = independentAgent([PGagent(agentParam) for i in range(n_agents)])
    multiPG = Agents([IAC_RNN(9,675,device=device) for i in range(n_agents)])  # create PGagents as well as a social agent
    # multiPG = Social_Agents([social_IAC(8,400,agentParam) for i in range(n_agents)],agentParam)
    for i_episode in range(101):
        n_state, ep_reward = env.reset(), 0  # reset the env
        for t in range(1, 3000):
            actions = multiPG.choose_action(n_state)  # agent.select_action(state)   #select masked actions for every agent
            # actions = multiPG.select_masked_actions(n_state)
            n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            if args.render:  # render or not
                env.render()
            # plt.close()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward += sum(n_reward.values())  # record the total reward
            multiPG.update(n_state.values(), n_reward.values(), n_state_.values(), actions.values())
            # multiPG.update_law()
            n_state = n_state_

        running_reward = ep_reward
        # loss = multiPG.update_agents()  # update the policy for each PGagent
        # multiPG.update_law()  # update the policy of law
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            logger.scalar_summary("ep_reward", ep_reward, i_episode)

        # if i_episode % save_eps == 0 and i_episode > 1 and ifsave_model:
        #     multiPG.save(file_name)
        # #


def MAAC_main():
    n_agents = 4
    n_rollout_threads = 12
    env = make_parallel_env(n_rollout_threads,3)
    agent_init_params = [{'num_in_pol': 675, 'num_out_pol': 9} for i in range(n_agents)]
    multiPG = AttentionSAC(agent_init_params, [[675,9] for a in range(n_agents)])  # create PGagents as well as a social agent
    multiPG = AttentionSAC.init_from_env(env,
                                       tau=0.001,
                                       pi_lr=0.001,
                                       q_lr=0.001,
                                       gamma=0.99,
                                       pol_hidden_dim=128,
                                       critic_hidden_dim=128,
                                       attend_heads=4,
                                       reward_scale=100.)
    replay_buffer = ReplayBuffer(10000,4,[675,675,675,675],[9,9,9,9])
    for i_episode in range(101):
        print("i_episode:",i_episode)
        n_state, ep_reward1, ep_reward2 = env.reset(), 0, 0  # reset the env
        # n_state2,ep_reward2 = env2.reset(), 0
        for t in range(1, 3000):
            # n_state = np.array(list(n_state.values())).reshape(675,-1)
            # n_state2 = np.array(list(n_state2.values())).reshape(675, -1)
            # i1 = n_state[:,1]
            # i2 = np.vstack(n_state[:,1])
            print(t)
            torch_obs = [torch.autograd.Variable(torch.Tensor(np.vstack(n_state[:, i])),
                                  requires_grad=False)
                         for i in range(multiPG.nagents)]
            torch_agent_actions = multiPG.step(torch_obs,explore=True)  # agent.select_action(state)   #select masked actions for every agent
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
            # actions = multiPG.select_masked_actions(n_state)

            n_state_, n_reward, done, _ = env.step(actions)  # interact with the env
            # if args.render:  # render or not
            #     env.render()
            # plt.close()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward1 += sum(n_reward[0])  # record the total reward
            ep_reward2 += sum(n_reward[1])
            replay_buffer.push(n_state, agent_actions, n_reward, n_state_, done)
            n_rollout_threads = 12
            t += n_rollout_threads
            use_gpu = False
            if (len(replay_buffer) >= 1024 and
                    (t % 100) < 12):
                if use_gpu:
                    multiPG.prep_training(device='gpu')
                else:
                    multiPG.prep_training(device='cpu')
                for u_i in range(4):
                    sample = replay_buffer.sample(1024,
                                                  to_gpu=use_gpu)
                    multiPG.update_critic(sample, logger=None)
                    multiPG.update_policies(sample, logger=None)
                    multiPG.update_all_targets()
                multiPG.prep_rollouts(device='cpu')
            # multiPG.update_law()
            n_state = n_state_

        # running_reward = ep_reward
        # loss = multiPG.update_agents()  # update the policy for each PGagent
        # multiPG.update_law()  # update the policy of law
        if i_episode % args.log_interval == 0:
            print('Episode {}\tAverage reward 1: {:.2f}\tAverage reward 2: {:.2f}'.format(
                i_episode, ep_reward1, ep_reward2))
            # logger.scalar_summary("ep_reward", ep_reward, i_episode)

        # if i_episode % save_eps == 0 and i_episode > 1 and ifsave_model:
        #     multiPG.save(file_name)
        # #


if __name__ == '__main__':
    MAAC_main()
