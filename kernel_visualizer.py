import numpy as np


def gaus(x, h):
    return 1 / (h * 2 * np.pi) * np.exp(-np.square(x / h))

def gaus2(x, sig):
    return np.exp(-np.square(x / sig))

x = np.linspace(0, 10, 1000)
import matplotlib.pyplot as plt
plt.plot(x, gaus(x, 5))
#plt.plot(x, 1/np.sqrt(x/200))
plt.show()
