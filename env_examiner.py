#from pyvirtualdisplay import Display
#
#display = Display(visible=0, size=(80, 60))
#display.start()

import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from PIL import Image

from cartpole_wrapper import pixels_cropped_wrapper

env = gym.make("CartPole-v0")
print(f"original env: {env.observation_space}")

env = pixels_cropped_wrapper(env, diff=True)
print(f"wrapped env: {env.observation_space}")

obv = env.reset()
print(obv.shape)
for _ in range(10):
    obv, *_ = env.step(1)

plt.imshow(obv)
plt.show()

env.close()
