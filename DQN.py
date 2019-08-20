import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_dqn import DQNExternalMem

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = DQNExternalMem(MlpPolicy, env, verbose=1, tensorboard_log="./logs",)


from stable_baselines.gail import ExpertDataset
dataset = ExpertDataset(expert_path='mfec_expert_cartpole.npz',
                        traj_limitation=1000, batch_size=128)
model.pretrain(dataset, n_epochs=1000)

model.learn(total_timesteps=250000)
model.save("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
