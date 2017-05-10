from attention_board import AttentionBoard
from board_display import *

from PIL import Image

import numpy as np
import sys
import math

SIZE=84

class AttentionEnv:
    metadata = {'render.modes': ['human', 'array']}

    def __init__(self, complex=False, sum_reward=False):
        if complex:
            self.board = AttentionBoard2(SIZE)
        else:
            self.board = AttentionBoard(SIZE)
        self.display = None
        self.reward = 0.
        self._sum_reward = sum_reward

    def step(self, action):
        step_reward = self.board.step(action)
        if self._sum_reward:
            self.reward += step_reward
        else:
            self.reward = step_reward
        obs = self.board.next()
        return (obs, self.reward, False, None)

    def reset(self):
        self.board = AttentionBoard(SIZE)
        self.reward = 0.
        return self.board.image()

    def render(self, mode='human', close=False):
        if mode == 'human':
            if self.display == None:
                self.display = BoardDisplay(SIZE, SIZE)
            self.display.render_update(self.board)
        return self.board.image()
