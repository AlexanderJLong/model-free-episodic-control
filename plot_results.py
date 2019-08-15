import matplotlib
import matplotlib.pyplot as plt

from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns

"""
structure is:
time, frames, episodes, reward_avg, reward_max

filenames are: ..._K=1_SEED=1
 """


def plot_data(data, xaxis='rounded_frames', value="reward_avg", condition="Condition1", smooth=1, n=1, compare="k", **kwargs):
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
    sns.set(style="darkgrid", font_scale=1.5)
    plot = sns.lineplot(data=data,
                 x=xaxis,
                 y=value,
                 ci='sd',
                 estimator=np.mean,
                 hue=compare,
                 palette=sns.color_palette("Set1", n),
                 **kwargs)
    axes = plot.axes
    plot.set_title(TITLE)
    axes.set_xlim(0, )
    plt.tight_layout(pad=0.5)
    plt.xlim(0, None)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()

data = []
TITLE = "Noautonorm"
if TITLE:
    base_dirs = glob("./agents/" + TITLE + "*SEED=1*/")
    base_dirs = glob("./agents/" + TITLE + "*SEED=1*/")
else:
    base_dirs = glob("./agents/*SEED=1*/")
for i in range(0, len(base_dirs)):
    base_dir = base_dirs[i][:-2]  # get current run and strip off the seed
    files = glob(base_dir + "*/results.csv")
    print(files)

    for f in files:
        table = pd.read_csv(f, sep=',', header=0)
        f = f.split("/")[-2]

        for param in f.split("_"):
            if "=" in param:
                p, v = param.split("=")
                table.insert(len(table.columns), p, v)
        data.append(table)

print(data)
plot_data(data, smooth=1, n=len(base_dirs), compare=None)
plt.show()
plt.savefig("./plots/"+TITLE+".png")
