import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_dqn import DQNExternalMem

import tensorflow as tf

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = DQNExternalMem(LnMlpPolicy, env,
                       gamma=0.99,
                       verbose=1,
                       tensorboard_log="./logs",
                       #prioritized_replay=True,
                       #target_network_update_freq=2000,
                       #exploration_fraction=1e-5,
                       #param_noise=True,
                       #learning_rate=1e-3,
                       #prioritized_replay=True,
                       exploration_final_eps=0,
                       batch_size=64,
                       exploration_fraction=0.3,
                       policy_kwargs=dict(act_fun=tf.nn.relu, layers=[32, 32]))


#from stable_baselines.gail import ExpertDataset
#dataset = ExpertDataset(expert_path='mfec_expert_cartpole.npz',
#                        traj_limitation=-1, batch_size=50)
#model.pretrain(dataset, n_epochs=20)
#
model.learn(total_timesteps=80000)
model.save("deepq_cartpole")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
