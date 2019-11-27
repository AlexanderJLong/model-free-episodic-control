from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



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
g = sns.FacetGrid(df, col="ENV", hue="WEIGHTING", col_wrap=4, sharey=False)
(g.map(sns.lineplot, "rounded_frames", "reward_avg", ci='sd', estimator=np.mean, )).set_titles("{col_name}")
g.add_legend()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig(f"./plots/full_run.png")
plt.show()
