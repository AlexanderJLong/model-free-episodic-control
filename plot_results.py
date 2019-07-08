import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns

result_files = glob("./agents/*/results.csv")
"""
structure is:
time, frames, episodes, reward_avg, reward_max
 """

data = []

for r in result_files:
    data = pd.read_csv(r, sep=',', header=0)

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
        xscale = np.max(np.asarray(data['frames'])) > 5e3
        if xscale:
            # Just some formatting niceness: x-axis scale in scientific notation if max x is large
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.tight_layout(pad=0.5)
        title = r.split("/")[2]+"-smoothed"+str(smooth)
        plt.title(title)
        plt.savefig("plots/" + title)
        print(r)
        plt.show()


exit()
