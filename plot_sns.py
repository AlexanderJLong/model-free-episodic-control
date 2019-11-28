from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# simple, otrainbow at 100k
sota = {
    "alien": (405.2, 739.9),
    "amidar": (88, 188.6),
    "assault": (369.3, 431.2),
    "asterix": (1089.5, 470.8),
    "bank_heist": (8.2, 51.0),
    "battle_zone": (5184.4, 10124.6),
    "boxing": (9.1, 0.2),
    "breakout": (12.7, 1.9),
    "chopper_command": (1246.9, 861.8),
    "crazy_climber": (398217.8, 16185.3),
    "demon_attack": (169.5, 508.0),
    "freeway": (20.3, 27.9),
    "frostbite": (254.7, 886.8),
    "gopher": (771.0, 349.5),
    "hero": (1295.1, 6857.0),
    "jamesbond": (125.3, 301.6),
    "kangaroo": (323.1, 779.3),
    "krull": (4539.9, 2851.5),
    "kung_fu_master": (17257.2, 14346.1),
    "ms_pacman": (761.8, 1204.1),
    "pong": (5.2, 19.3),
    "private_eye": (58.3, 97.8),
    "qbert": (559.8, 1152.9),
    "road_runner": (5169.4, 9600.0),
    "seaquest": (370.9, 354.1),
    "up_n_down": (2152.6, 2877.4),
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

print(df.to_string())
sns.set_context("paper")
sns.set(style="darkgrid")
g = sns.FacetGrid(df, col="ENV", hue="DISTANCE", col_wrap=4, sharey=False)
(g.map(sns.lineplot, "rounded_frames", "reward_avg", ci='sd', estimator=np.mean, )).set_titles("{col_name}")

max_frames = max(df["rounded_frames"])
for ax in g.axes.flat:
    env_name = ax.get_title()
    #ax.plot((0, max_frames), (sota[env_name][0], sota[env_name][0]), c="k", linewidth=1, ls=":", label="SimPLe")
    ax.plot((0, max_frames), (sota[env_name][1], sota[env_name][1]), c="k", linewidth=1, ls="--", label="Rainbow (OT)")
plt.legend()
#g.add_legend()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig(f"./plots/full_run.png")
plt.show()
