#from pyvirtualdisplay import Display
#
#display = Display(visible=0, size=(80, 60))
#display.start()

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from PIL import Image

from cartpole_wrapper import pixel_state_wrapper

env = gym.make("CartPole-v0")
print(f"original env: {env.observation_space}")

env = pixel_state_wrapper(env)
print(f"wrapped env: {env.observation_space}")

obv = env.reset()
print(obv.shape)
for _ in range(10):
    obv, *_ = env.step(1)

plt.imshow(obv)
plt.show()

env.close()
