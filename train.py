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
model = DQN(NUM_ACTIONS)
memory = ReplayMemory(REPLAY_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

if len(sys.argv) == 3:
    load_path = sys.argv[2]
    model.load_state_dict(torch.load(load_path))

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
        return torch.LongTensor([[random.randrange(NUM_ACTIONS)]])

def plot_durations():
    plt.figure(REWARD_GRAPH)
    plt.clf()
    durations_t = torch.Tensor(episode_durations)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
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
learning_starts = REPLAY_SIZE /2 
for i_episode in range(EPOCHS * EPOCH_SIZE):
    # Initialize the environment and state
    frame = preprocess(env.reset())
    states = deque([frame] * 4)

    state = torch.stack(states, dim=0).unsqueeze(0)
    num_steps = 0
    total_reward = 0
    for t in range(MAX_TIME+1):
        if RENDER: env.render()

        # Select and perform an action
        if t > learning_starts:
            action = select_action(state)
        else:
            action = torch.LongTensor([[random.randrange(NUM_ACTIONS)]])

        next_frame, reward, done, _ = env.step(action.numpy())

        reward = max(-1.0, min(reward, 1.0))
        total_reward += reward
        
        reward = torch.FloatTensor([reward])

        # Observe new state
        states.popleft()
        states.append(preprocess(next_frame))
        next_state = torch.stack(states, dim=0).unsqueeze(0)

        if done:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if t > learning_starts:
            optimize_model()
        if done or t == MAX_TIME:
            num_steps=t
            episode_durations.append(total_reward)
            plot_durations()
            break
    
    if i_episode % EPOCH_SIZE == 0:
        epoch = i_episode / EPOCH_SIZE
        print "Epoch: {0} // Reward: {1} // Num Steps: {2}".format(epoch, total_reward, num_steps)
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
        torch.save(model.state_dict(), model_path)
        plt.savefig(figure_path)

        if HINTON:
            print_weights(model, epoch, filename)
        

env.close()
plt.ioff()
if RENDER:
    plt.show()