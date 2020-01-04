from rainbow_env import EnvStacked
env = EnvStacked(seed=1, game="ms_pacman", sticky_actions=False, stacked_states=4)

from tqdm import tqdm

done = True

for i in tqdm(range(100_000)):
    if done:
        s = env.reset()
    R = 0
    s, r, done = env.step(0)
    R += r