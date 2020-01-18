import numpy as np
def gaus(x, sig):
    return np.exp(-np.square(x / sig) / 2)

def gaus2(x, sig):
    return np.exp(-np.square(x / sig))

x = np.linspace(0, 10000, 1000)
import matplotlib.pyplot as plt
plt.plot(x, gaus(x, 1000))
plt.plot(x, gaus2(x, 1500))
#plt.plot(x, 1/np.sqrt(x/200))
plt.show()
