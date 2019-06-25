import roboschool
import gym

import matplotlib.pyplot as plt

from pyvirtualdisplay import Display

display = Display(visible=0, size=(80, 60))
display.start()


from cartpole_wrapper import pixel_state_wrapper
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env = pixel_state_wrapper(env)

obv = env.reset()
print(obv)
plt.imshow(obv)
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
