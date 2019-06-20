import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import matplotlib
from matplotlib.pyplot import figure
result_files = glob("./agents/*/results.csv")
"""
structure is:
time, frames, episodes, reward_avg, reward_max
 """


data=[]

for r in result_files:
    data.append(np.genfromtxt(r, delimiter=",")[1:])

f, axs = plt.subplots(2, 2, sharey=False)
from scipy.ndimage.filters import gaussian_filter1d

plt.ticklabel_format(style='sci')
for i, ax in enumerate(axs.flatten()):
    ax.plot(data[i][:,1], data[i][:,3], alpha=0.15)
    ax.set_title(result_files[i].split("/")[2])

    ysmoothed = gaussian_filter1d(data[i][:,3], sigma=15)
    ax.plot(data[i][:,1], ysmoothed)


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig("plot")
plt.show()

print(data[0][:,0])