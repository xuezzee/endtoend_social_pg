"""
Coin Game environment.
"""
import gym
import numpy as np
import tkinter
from gym.spaces import Discrete, Tuple
# from gym.spaces import prng


class CoinGameVec(gym.Env):
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    def __init__(self, max_steps, batch_size, grid_size=2):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None
        self.root = None

    def reset(self):
        self.step_count = 0
        self.red_coin = np.random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = np.random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = np.random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = np.random.randint(self.grid_size, size=2)
            self._generate_coin(i)
        state = self._generate_state()
        state = np.reshape(state, (self.batch_size, -1))
        observations = [state, state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observations, info

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agents
        success = 0
        while success < 2:
            success = 0
            self.coin_pos[i] = np.random.randint(self.grid_size, size=(2))
            success = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        self.map = state
        return state

    def step(self, actions):
        test_reward = 0
        ac0, ac1 = actions

        self.step_count += 1

        for j in range(self.batch_size):
            a0, a1 = ac0[j], ac1[j]
            assert a0 in {0, 1, 2, 3} and a1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[a0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[a1]) % self.grid_size

        # Compute rewards
        reward_red, reward_blue = np.zeros(self.batch_size), np.zeros(self.batch_size)
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    test_reward += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -2  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    reward_blue[i] += 1
                    test_reward += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    test_reward += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1
                    test_reward += 1

            if generate:
                self._generate_coin(i)

        reward = [reward_red, reward_blue]
        state = self._generate_state().reshape((self.batch_size, -1))
        observations = [state, state]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, reward, done, info, test_reward

    def _close_view(self):
        if self.root:
            self.root.destory()
            self.root = None
            self.canvas = None
        # self.done = True

    def render(self):
        scale = 30
        width = self.map.shape[2] * scale
        height = self.map.shape[3] * scale
        if self.root is None:
            self.root = tkinter.Tk()
            self.root.title("social_dilemmas")
            self.root.protocol("WM_DELETE_WINDOW", self._close_view)
            self.canvas = tkinter.Canvas(self.root, width=width, height=height)
            self.canvas.pack()

        self.canvas.delete(tkinter.ALL)
        self.canvas.create_rectangle(0, 0, width, height, fill="black")

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * scale,
                y * scale,
                (x + 1) * scale,
                (y + 1) * scale,
                fill=color
            )

        colors = ["Cyan","Orange","Blue","Red"]
        for i in range(self.map.shape[1]):
            color = colors[i]
            for x in range(self.map.shape[2]):
                for y in range(self.map.shape[3]):
                    if self.map[0,i,x,y] == 1:
                        fill_cell(x,y,color)

        # for x in range(map.shape[0]):
        #     for y in range(map.shape[1]):
        #         if map[x,y] == '@':
        #             fill_cell(x,y,'Grey')
        #         if map[x,y] == 'A':
        #             fill_cell(x,y,'Green')
        #         if map[x,y] == 1.:
        #             fill_cell(x,y,'Red')
        #         if map[x,y] == 2.:
        #             fill_cell(x,y,'Blue')
        #         if map[x,y] == 'H':
        #             fill_cell(x,y,'Orange')
        #         if map[x,y] == 'S':
        #             fill_cell(x,y,'Navy')
        #         if map[x,y] == 'R':
        #             fill_cell(x,y,'Cyan')
        #         if map[x,y] == 'F':
        #             fill_cell(x,y,'Yellow')
        #         if map[x,y] == 'C':
        #             fill_cell(x,y,'Pink')
        #         if map[x,y] == 'B':
        #             fill_cell(x,y,'Pink')
        #         if map[x,y] == 'p':
        #             fill_cell(x,y,'Purple')
        #         if map[x,y] == 3.:
        #             fill_cell(x,y,'Magenta')
        #         if map[x,y] == 4.:
        #             fill_cell(x,y,'Lavender')
        #         if map[x,y] == '5':
        #             fill_cell(x,y,'Purple')


        self.root.update()