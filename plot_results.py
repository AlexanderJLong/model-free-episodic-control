import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import re

"""
structure is:
time, frames, episodes, reward_avg, reward_max

filenames are: ..._K=1_SEED=1
 """


def plot_data(data, xaxis='rounded_frames', value="reward_avg", condition="Condition1", smooth=1, n=1, **kwargs):
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
                 hue="K",
                 palette=sns.color_palette("Set1", n),
                 **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:
    Changes the colorscheme and the default legend style, though.
    """
    # plt.legend(loc='best').set_draggable(True)

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """
    axes = plot.axes
    plot.set_title(TITLE)
    axes.set_xlim(0, )
    plt.tight_layout(pad=0.5)
    plt.xlim(0, None)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


data = []
TITLE = "*"
if TITLE:
    base_dirs = glob("./agents/*" + TITLE + "*SEED=1*/")
else:
    base_dirs = glob("./agents/*SEED=1*/")
for i in range(0, len(base_dirs)):
    base_dir = base_dirs[i][:-2]  # get current run and strip off the seed
    files = glob(base_dir + "*/results.csv")
    print(files)

    for f in files:
        table = pd.read_csv(f, sep=',', header=0)
        table.insert(len(table.columns), 'K', re.findall(r'\d+', f)[-2])
        table.insert(len(table.columns), 'SEED', re.findall(r'\d+', f)[-1])
        table.insert(len(table.columns), 'DIM', re.findall(r'\d+', f)[-3])
        data.append(table)

print(data)
plot_data(data, smooth=1, n=len(base_dirs))
plt.show()
plt.gcf()
#Not working
plt.savefig("./plots/"+TITLE+".png")
