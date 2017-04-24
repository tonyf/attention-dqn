from gym_attention.envs import *
import numpy as np
env = AttentionEnv()
env.reset()

while True:
    env.step(np.random.randint(80, size=2))
    env.render()
