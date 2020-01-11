import numpy as np
def gaus(x, sig):
    return np.exp(-np.square(x / sig) / 2)

x = np.linspace(0, 10000, 1000)
import matplotlib.pyplot as plt
plt.plot(x, gaus(x, 7000))
plt.show()
