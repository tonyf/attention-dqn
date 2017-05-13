import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import scipy.misc

from dqn import *
from replay import *
from hinton import *
from visualize import *

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
env.reset()
plt.figure()
plt.show()


BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
USE_CUDA = torch.cuda.is_available()

model = DQNFF()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters(), lr=0.005)

if USE_CUDA:
    model.cuda()

COMPLEX=False
if len(sys.argv) == 2:
    load_path = sys.argv[1]
    model.load_state_dict(torch.load(load_path))
    print "Loaded weights from {0}".format(load_path)
    COMPLEX=True


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def print_weights(model, episode, filename):
    print "Saving weight diagrams"
    fc1_weights = model.fc1.state_dict()['weight'].numpy()
    fc2_weights = model.head.state_dict()['weight'].numpy()

    visualize_weights(fc1_weights, "FC1", episode, filename)
    visualize_weights(fc2_weights, "FC2", episode, filename)
    print "Exported diagrams"

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    # print eps_threshold
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.LongTensor([[random.randrange(2)]])


episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.Tensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

last_sync = 0
def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.ByteTensor(
        tuple(map(lambda s: s is not None, batch.next_state)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
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

num_episodes = 5000
saved_half = False
saved_full = False
saved_overtrained = False
for i_episode in range(num_episodes):
    # Initialize the environment and state
    last_frame = env.reset()
    current_frame = last_frame
    state = torch.from_numpy(current_frame - last_frame).float().unsqueeze(0)
    duration = 0
    for t in count():
        # env.render()
        last_frame = current_frame
        action = select_action(state)
        current_frame, reward, done, _ = env.step(action.numpy()[0][0])
        reward = torch.Tensor([reward])

        # Observe new state
        next_state = torch.from_numpy(current_frame - last_frame).float().unsqueeze(0)
        if done: next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            duration = t+1
            episode_durations.append(duration)
            plot_durations()
            break
    mean_duration = np.mean(episode_durations[-100:])
    print "Episode: {0} // Duration: {1} // Mean Duration: {2}".format(i_episode, duration, mean_duration)

    path = ""
    if COMPLEX: 
        path += "complex_"

    if mean_duration >= 300:
        path += "over_trained"
        filename = "models/" + path
        torch.save(model.state_dict(), filename)
        if not saved_overtrained:
            print_weights(model, i_episode, path)
            saved_overtrained = True
        break
    elif mean_duration >= 195:
        path += "fully_trained"
        filename = "models/" + path
        torch.save(model.state_dict(), filename)
        if not saved_full:
            print_weights(model, i_episode, path)
            saved_full = True
    elif mean_duration >= 98:
        path += "half_trained"
        filename = "models/" + path
        torch.save(model.state_dict(), filename)
        if not saved_half:
            print_weights(model, i_episode, path)
            saved_half = True

    duration = 0

plt.savefig("figures/cartpole")
env.close()
plt.ioff()
# plt.show()

