import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

from PIL import Image
#from pyvirtualdisplay import Display
#
#display = Display(visible=0, size=(80, 60))
#display.start()

from cartpole_wrapper import pixel_state_wrapper
from gym.utils.play import play


env = gym.make("CartPole-v0")

print(f"original env: {env.observation_space}")
env = pixel_state_wrapper(env, greyscale=True, difference=True, scale=False)
print(f"wrapped env: {env.observation_space}")

print(env.observation_space)
#play(env, keys_to_action={(ord('a'),): 1, (ord('s'),): 0})


obv1 = env.reset()
print(obv1.shape)
im = Image.fromarray(obv1)
#3im.thumbnail((84, 84), Image.BICUBIC)
#im.show()
plt.imshow(np.asarray(im))

plt.imshow(obv1)

plt.show()

obv = env.reset()
obv = env.reset()
obv = env.reset()
obv = env.reset()
plt.imshow(obv1-obv)
plt.show()

for _ in range(10):
    obv, *_ = env.step(1)
plt.imshow(obv)
for l in obv:
    print(l)
plt.show()

env.close()
