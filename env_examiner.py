import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt


from dopamine_env import create_atari_environment


env = create_atari_environment("MsPacman")


print(env.observation_space)
#play(env, keys_to_action={(ord('a'),): 1, (ord('s'),): 0})

def show_stack(obv):
    for i, f in enumerate(obv):
        print(f"frame {i}")
        plt.imshow(f, cmap="Greys")
        plt.show()

obv1 = env.reset()
print(obv1.shape)
show_stack(obv1)
plt.show()

obv = env.reset()
show_stack(obv1)

plt.show()

for _ in range(100):
    obv, *_ = env.step(1)
show_stack(obv)

plt.imshow(obv[-1] - obv[-2])
plt.show()


env.close()
