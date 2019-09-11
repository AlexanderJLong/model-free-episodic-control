import numpy as np
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


from pyvirtualdisplay import Display
display = Display(visible=0, size=(80, 60))
display.start()

env_name = 'Pendulum-v0'
nproc = 4
T = 200

from cartpole_wrapper import pixels_cropped_wrapper

env = gym.make("CartPole-v0")
env = pixels_cropped_wrapper(env, diff=True)

def make_env(seed):
    def _f():
        env = gym.make("CartPole-v0")
        env = pixels_cropped_wrapper(env, diff=True)
        env.seed(seed)
        return env
    return _f

envs = [make_env(seed) for seed in range(nproc)]
envs = SubprocVecEnv(envs)

xt = envs.reset()
states = []
actions = []
rewards = []
dones = []
for t in range(T):
    ut = np.stack([0 for _ in range(nproc)])
    st, r, d, _ = envs.step(ut)
    states.append([s.flatten() for s in st])
    actions.append(ut)
    rewards.append(r)
    dones.append(d)

states=np.asarray(states)
print(states)
for s,a,r,d in zip(states, actions,rewards, dones):
    print(s,a,r,d)