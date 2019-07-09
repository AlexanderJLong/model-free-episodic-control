import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns

base_dir = glob("./agents/*SEED_1*/")
base_dir = base_dir[-1][:-2] #get last run and strip off the seed
"""
structure is:
time, frames, episodes, reward_avg, reward_max

filenames are: ..._SEED_1
 """
def plot_data(data, xaxis='rounded_frames', value="reward_avg", condition="Condition1", smooth=1, **kwargs):
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
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    print(data)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, ci='sd', estimator=np.mean, **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:
    Changes the colorscheme and the default legend style, though.
    """
    #plt.legend(loc='best').set_draggable(True)

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()
plt.tight_layout(pad=0.5)

data = []

files = glob(base_dir+"*/results.csv")
print(files)


for f in files:
    data.append(pd.read_csv(f, sep=',', header=0))


plot_data(data, smooth=1)
exit()
for smooth in [1, 10]:
    if smooth > 1:
        y = np.ones(smooth)
        x = np.asarray(data["reward_avg"])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        try:
            data["reward_avg"] = smoothed_x
        except ValueError:
            continue
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x="frames", y="reward_avg", ci='sd')
    xscale = np.max(np.asarray(data['episodes'])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)
    title = base_dir.split("/")[2]+"-smoothed"+str(smooth)
    plt.title(title)
    plt.savefig("plots/" + title)
    print(base_dir)
    plt.show()


exit()
