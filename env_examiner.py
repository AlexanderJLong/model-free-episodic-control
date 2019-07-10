import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

#from pyvirtualdisplay import Display
#
#display = Display(visible=0, size=(80, 60))
#display.start()

from cartpole_wrapper import pixel_state_wrapper


env = gym.make("CartPole-v0")

print(f"original env: {env.observation_space}")
env = pixel_state_wrapper(env)
print(f"wrapped env: {env.observation_space}")

from gym.utils.play import play
#play(env, keys_to_action={(ord('a'),): 1, (ord('s'),): 0})

obv1 = env.reset()
plt.imshow(obv1)
plt.show()

obv = env.reset()
plt.imshow(obv1-obv)
plt.show()

obv, *_ = env.step(0)
obv, *_ = env.step(0)
obv, *_ = env.step(0)
obv, *_ = env.step(0)
obv, *_ = env.step(0)
obv, *_ = env.step(0)
print(obv)
plt.imshow(obv)
plt.show()

env.close()
