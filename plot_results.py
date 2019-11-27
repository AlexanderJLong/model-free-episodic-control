from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("paper")
sns.set(style="darkgrid")

"""
structure is:
time, frames, episodes, reward_avg, reward_max

filenames are: ..._K=1_SEED=1
 """


def plot_data(data, env_name, xaxis='rounded_frames', value="reward_avg", smooth=1, compare="k", n=0):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    plot = sns.lineplot(data=data,
                        x=xaxis,
                        y=value,
                        ci='sd',
                        estimator=np.mean,
                        hue=compare,
                        palette=sns.color_palette("Set1", n),
                        sort=False,
                        )
    plot.set_title(env_name)
    plot.axes.set_xlim(0, )
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


env_dirs = glob("./agents/*SEED=1*/")
envs = sorted(list(set([d.replace("=", ":").split(":")[1] for d in env_dirs])))
print(envs)

if len(envs) > 1:
    ncols = len(envs) // 2
    nrows = len(envs) // ncols + 1
else:
    ncols = 1
    nrows = 1
#plt.subplots(constrained_layout=True)
for i, env in enumerate(envs):
    data = []
    plt.subplot(nrows, ncols, i + 1)
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
            data.append(table)
    print(len(base_dirs))

    plot_data(data, smooth=1, compare="WEIGHTING", env_name=env, n = len(base_dirs))

#plt.tight_layout()
plt.legend()
plt.savefig(f"./plots/{env}.png")
plt.show()
