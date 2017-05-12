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

if plt.isinteractive():
    plt.ioff()

env = AttentionEnv(complex=COMPLEX, sum_reward=SUM_REWARD, static=STATIC)

Q = DQN(env.action_space)
V = DQN(env.action_space)

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
if USE_CUDA:
    Q.cuda()
    V.cuda()

if len(sys.argv) == 3:
    load_path = sys.argv[2]
    Q.load_state_dict(torch.load(load_path))
    V.load_state_dict(torch.load(load_path))

memory = ReplayMemory(REPLAY_SIZE, frame_history_len=4)
optimizer = optim.RMSprop(Q.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
episode_rewards = []
num_updates=0

NUM_ACTIONS=env.action_space

class Variable(autograd.Variable):

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def select_action(state, t):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * t / EPS_DECAY)
    if sample > eps_threshold:
        state = torch.from_numpy(state).type(dtype).unsqueeze(0) / 255.0
        return Q(Variable(state, volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.IntTensor([[random.randrange(NUM_ACTIONS)]])

def plot_rewards(filename, rewards):
    plt.figure(num=REWARD_GRAPH)
    plt.clf()
    rewards_t = torch.Tensor(rewards)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(figure_path)

def optimize_model():
    global num_updates
    if not memory.can_sample(BATCH_SIZE):
        return
    state_batch, action_batch, reward_batch, next_state_batch, done_mask = memory.sample(BATCH_SIZE)

    # Compute a mask of non-final states and concatenate the batch elements
    # non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    # if USE_CUDA: non_final_mask = non_final_mask.cuda()
    # next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    # state_batch = Variable(torch.cat(batch.state))
    # action_batch = Variable(torch.cat(batch.action))
    # reward_batch = Variable(torch.cat(batch.reward))

    state_batch = Variable(torch.from_numpy(state_batch).type(dtype) / 255.0)
    action_batch = Variable(torch.from_numpy(action_batch).long())
    reward_batch = Variable(torch.from_numpy(reward_batch))
    next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype) / 255.0)
    not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions take
    # state_action_values = Q(state_batch).gather(1, action_batch)
    state_action_values = Q(state_batch).gather(1, action_batch.unsqueeze(1))
    # Compute V(s_{t+1}) for all next states.
    # next_state_values = Variable(torch.zeros(BATCH_SIZE))
    # next_state_values[non_final_mask] = V(non_final_next_states).detach().max(1)[0]
    # next_state_values.volatile = False
    next_state_values = V(next_state_batch).detach().max(1)[0]
    next_state_values = next_state_values * not_done_mask

    # Compute the expected Q values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)

    # Compute Bellman error
    bellman_error = expected_state_action_values - state_action_values
    clipped_bellman_error = bellman_error.clamp(-1, 1)
    d_error = clipped_bellman_error * -1.0

    # Optimize the model
    optimizer.zero_grad()
    state_action_values.backward(d_error.data.unsqueeze(1))
    optimizer.step()
    num_updates += 1

    # Periodically update the target network by Q network to target Q network
    if num_updates % V_UPDATE_FREQ == 0:
        V.load_state_dict(Q.state_dict())

def print_weights(model, epoch, filename):
    print "Saving weight diagrams"
    conv1_weights = np.reshape(model.conv1.state_dict()['weight'].numpy(), (128, 64))
    conv2_weights = np.reshape(model.conv2.state_dict()['weight'].numpy(), (2048, 16))
    fc1_weights = model.fc1.state_dict()['weight'].numpy()
    fc2_weights = model.fc2.state_dict()['weight'].numpy()

    visualize_weights(conv1_weights, "Conv1", epoch, filename)
    visualize_weights(conv2_weights, "Conv2", epoch, filename)

    visualize_weights(fc1_weights, "FC1", epoch, filename)
    visualize_weights(fc2_weights, "FC2", epoch, filename)
    print "Exported diagrams"

print "Start training"
state = env.reset()
episode_reward = 0
num_steps = 0
for t in count():
    if RENDER: env.render()

    state_idx = memory.store_frame(state)
    stacked_state = memory.encode_recent_observation()

    # Select and perform an action
    if t > START_TRAINING:
        action = select_action(stacked_state, t)[0][0]
    else:
        action = random.randrange(NUM_ACTIONS)

    next_state, reward, done, _ = env.step(action)
    reward = max(-1.0, min(reward, 1.0))
    episode_reward += reward
    num_steps += 1

    memory.store_effect(state_idx, action, reward, done)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the target network)
    if t > START_TRAINING and t % EPOCH_SIZE == 0: 
        optimize_model()
    if done:
        episode_rewards.append(episode_reward)
        episode_reward=0
        num_steps=0
        next_state = env.reset()
    state = next_state
    
    if t % EPOCH_SIZE == 0:
        epoch = t / EPOCH_SIZE
        print "Epoch: {0} // Reward: {1} // Num Steps: {2}".format(epoch, episode_reward, num_steps)
        # save model
        filename = 'simple_'
        if COMPLEX:
            filename = "complex_"
        if STATIC:
            filename = "static_"
        if SUM_REWARD:
            filename= filename + "sum_"
        filename = filename + str(epoch)

        model_path = "models/" + filename
        figure_path = "figures/" + filename
        torch.save(Q.state_dict(), model_path)
        plot_rewards(figure_path, episode_rewards)

        if HINTON: print_weights(Q, epoch, filename)
        
if RENDER: plt.show()