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

env = AttentionEnv()
model = DQN(9)
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())

episode_durations = []
last_sync = 0
steps_done = 0

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    model.cuda()


class Variable(autograd.Variable):

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.LongTensor([[random.randrange(9)]])

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.Tensor(episode_durations)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

def get_screen():
    screen = torch.from_numpy(env.render()).float()
    return screen

def preprocess(frame):
    frame = frame / 255
    return torch.from_numpy(frame).float()

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]), volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions take
    # In this case, since 
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# env.reset()
# plt.figure()
# get_screen()
# plt.title('Example extracted screen')
# plt.show()
#
# while True:
#     env.step(np.random.randint(80, size=2))
#     env.render()

print "Start training"
num_episodes = 10
for i_episode in range(EPOCHS * EPOCH_SIZE):
    # Initialize the environment and state
    final_reward = 0
    frame = preprocess(env.reset())
    states = deque([frame] * 4)

    state = torch.stack(states, dim=0).unsqueeze(0)
    for t in range(MAX_TIME):
        if RENDER: env.render()

        # Select and perform an action
        action = select_action(state)
        next_frame, reward, _, _ = env.step(action.numpy())
        final_reward = reward
        reward = torch.FloatTensor([reward])

        # Observe new state
        states.popleft()
        states.append(preprocess(next_frame))
        next_state = torch.stack(states, dim=0).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if t == MAX_TIME-1:
            episode_durations.append(t + 1)
            plot_durations()
            break
    
    print "Episode: {0} // Reward: {1}".format(i_episode, final_reward)
    if i_episode % EPOCH_SIZE == 0:
        # save model
        torch.save(model.state_dict(), "models/simple_{0}".format(i_episode))
        plt.savefig("figures/simple_{0}".format(i_episode))

env.close()
plt.ioff()
if RENDER:
    plt.show()