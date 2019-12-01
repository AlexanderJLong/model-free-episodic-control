from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# simple, otrainbow at 100k, human
sota = {
    "alien": (405.2, 739.9, 7127),
    "amidar": (88, 188.6, 1719),
    "assault": (369.3, 431.2, 742),
    "asterix": (1089.5, 470.8, 8503),
    "bank_heist": (8.2, 51.0, 753),
    "battle_zone": (5184.4, 10124.6, 37187),
    "boxing": (9.1, 0.2, 12),
    "breakout": (12.7, 1.9, 30),
    "chopper_command": (1246.9, 861.8, 7387),
    "crazy_climber": (398217.8, 16185.3, 35829),
    "demon_attack": (169.5, 508.0, 1971),
    "freeway": (20.3, 27.9, 29),
    "frostbite": (254.7, 886.8, 4334),
    "gopher": (771.0, 349.5, 2412),
    "hero": (1295.1, 6857.0, 30826),
    "jamesbond": (125.3, 301.6, 302),
    "kangaroo": (323.1, 779.3, 3035),
    "krull": (4539.9, 2851.5, 2665),
    "kung_fu_master": (17257.2, 14346.1, 22736),
    "ms_pacman": (761.8, 1204.1, 6951),
    "pong": (5.2, 19.3, 14),
    "private_eye": (58.3, 97.8, 69571),
    "qbert": (559.8, 1152.9, 13455),
    "road_runner": (5169.4, 9600.0, 7845),
    "seaquest": (370.9, 354.1, 42054),
    "up_n_down": (2152.6, 2877.4, 11693),
}

"""
structure is:
time, frames, episodes, reward_avg, reward_max

filenames are: ..._K=1_SEED=1
 """
env_dirs = glob("./agents/*SEED=1*/")
envs = sorted(list(set([d.replace("=", ":").split(":")[1] for d in env_dirs])))
data = []
df = pd.DataFrame()
for i, env in enumerate(envs):
    data = []
    base_dirs = glob(f"./agents/ENV={env}*SEED=1*/")
    print(env)
    try:
        for bd in base_dirs:
            base_dir = bd[:-4]  # get current run and strip off the seed
            files = glob(base_dir + "*/results.csv")
            for f in files:
                table = pd.read_csv(f, sep=',', header=0)
                f = f.split("/")[-2]

                for param in f.split(":"):
                    if "=" in param:
                        p, v = param.split("=")
                        if p == "SEED":
                            print(f"seed {v}")
                        table.insert(len(table.columns), p, v)
                df = pd.concat([df, table], ignore_index=True)
        print(len(base_dirs))
    except:
        continue

""""
print(df.to_string())
sns.set_context("paper")
sns.set(style="darkgrid")
g = sns.FacetGrid(df, col="ENV", hue="WEIGHTING", col_wrap=4, sharey=False)
(g.map(sns.lineplot, "rounded_frames", "reward_avg", ci='sd', estimator=np.mean, )).set_titles("{col_name}")

max_frames = max(df["rounded_frames"])
for ax in g.axes.flat:
    env_name = ax.get_title()
    if env_name in sota:
        # ax.plot((0, max_frames), (sota[env_name][0], sota[env_name][0]), c="k", linewidth=1, ls=":", label="SimPLe")
        ax.plot((0, max_frames), (sota[env_name][1], sota[env_name][1]), c="k", linewidth=1, ls="--",
                label="Rainbow (OT)")
"""

# hns
all_hns = []
timesteps = list(range(0, 100001, 10000))
print()
for env in sota.keys():
    print(env)
    means = []
    for t in timesteps:
        at_timestep = df['rounded_frames'] == t
        at_env = df['ENV'] == env
        average = df.loc[at_env & at_timestep, "reward_avg"].mean()
        means.append(average)
    means = np.asarray(means)
    hns = means/sota[env][2]
    if not np.isnan(hns[0]):
        all_hns.append(hns)


print(all_hns)
print(np.median(all_hns, axis=0))
plt.plot(timesteps, np.median(all_hns, axis=0))
plt.legend()
# g.add_legend()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig(f"./plots/full_run.png")
plt.show()
