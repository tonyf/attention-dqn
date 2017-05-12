import math
import random
import numpy as np

from collections import namedtuple, deque
from itertools import count
from copy import deepcopy
from PIL import Image

import matplotlib
from sys import platform as sys_pf
if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

from envs.attention_env import *
from replay import Transition, ReplayMemory
from dqn import DQN
from hyperparams import * 
from hinton import *

env = AttentionEnv(complex=COMPLEX, sum_reward=SUM_REWARD, static=STATIC)

Q = DQN(env.action_space)
V = DQN(env.action_space)

print "Loading Weights"
load_path = sys.argv[1]
Q.load_state_dict(torch.load(load_path))
V.load_state_dict(torch.load(load_path))

env.reset()
steps = 0
while True:
    env.render()
    state = torch.from_numpy(state).type(dtype).unsqueeze(0) / 255.0
    return Q(Variable(state, volatile=True)).data.max(1)[1].cpu()
    action = select_action(stacked_state, t)[0][0]
    _, _, done, _ = env.step(action)
    steps += 1
    if done:
        print "Done in {0} steps".format(steps)
        steps = 0
        env.reset()
