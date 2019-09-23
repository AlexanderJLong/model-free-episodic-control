import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from PIL import Image

from mfec.agent import cartpole_crop_grey_scale_normalize_resize
from cartpole_wrapper import pixels_cropped_wrapper

env = gym.make("Pong-v0")
print(f"original env: {env.observation_space}")

env = pixels_cropped_wrapper(env, crop=True, diff=False)
print(f"wrapped env: {env.observation_space}")

obv = env.reset()
print(obv.shape)
for _ in range(50):
    obv, *_ = env.step(1)

processed = cartpole_crop_grey_scale_normalize_resize(obv)
print(processed)
plt.imshow(processed)
plt.show()

env.close()
