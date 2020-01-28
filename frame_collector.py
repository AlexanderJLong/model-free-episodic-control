import cloudpickle as pkl
import numpy as np
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# load up the agent
with open("saves/agent.pkl", "rb") as f:
    agent = pkl.load(f)


from dopamine_env import create_atari_environment

env = create_atari_environment(
    game_name="Frostbite",
    sticky_actions=True,
    seed=0)

env.training = True
agent.training = False

ep_reward = 0
rewards = []
observation = env.reset()
observations = []
states = []

trace_id = 0
trace_ids = []
traces = []

for step in tqdm(list(range(50_000))):
    # Act, and record
    action, state, *_ = agent.choose_action(observation, step)
    observations.append(observation.flatten())
    states.append(state.flatten())
    trace_ids.append(trace_id)

    observation, reward, done, lifelost = env.step(action)
    ep_reward += reward
    if done:
        print(ep_reward)
        trace_id += 1
        rewards.append(ep_reward)
        ep_reward = 0
        observation = env.reset()


pkl.dump(states, open("saves/states.pkl", "wb"))


exit()
print(trace_id)
trace_ends = []  # index of last value in trace
for i, _ in enumerate(trace_ids[:-1]):
    if trace_ids[i] != trace_ids[i + 1]:
        trace_ends.append(i + 1)
trace_ends.append(len(trace_ids))

sns.set(context="paper", style="white")

reducer = umap.UMAP(n_neighbors=200)


def plot_traces(data, name):
    embedding = reducer.fit_transform(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=trace_ids)
    last_e = 0
    for e in trace_ends:
        plt.plot(embedding[last_e:e, 0], embedding[last_e:e, 1], alpha=0.3)
        last_e = e
    plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(f"plots/{name}.png")


plot_traces(observations, "obvs")
plot_traces(states, "states")
plt.show()
