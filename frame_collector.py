import cloudpickle as pkl
import numpy as np
from tqdm import tqdm
from rainbow_env import EnvStacked

#load up the agent
with open("saves/agent.pkl", "rb") as f:
    agent = pkl.load(f)

print('awd')
print([len(b) for b in agent.klt.buffers])

env = EnvStacked(
    seed=0,
    game="frostbite",
    sticky_actions=True,
    stacked_states=4)

env.training = False
agent.training = False

ep_reward = 0
rewards = []
observation = env.reset()
for step in tqdm(list(range(1_000))):
    # Act, and record
    action, state, Qs = agent.choose_action(observation)
    observation, reward, done = env.step(action)
    ep_reward += reward
    if done:
        print("done")
        rewards.append(ep_reward)
        ep_reward = 0
        observation = env.reset()

print(np.mean(rewards))
