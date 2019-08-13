from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import ACER

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env('PongNoFrameskip-v4', num_env=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)
model = ACER(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000)


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()