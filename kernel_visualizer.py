import numpy as np
def gaus(x, sig):
    return np.exp(-np.square(x / sig))

def l(x, sig):
    return np.exp(-(x / sig))
def gaus2(x, sig):
    return np.exp(-x / sig)

x = np.linspace(0, 6000, 1000)
import matplotlib.pyplot as plt
plt.plot(x, gaus(x, 1000))
plt.plot(x, l(x, 1000))
plt.plot(x, 1/np.sqrt(x))
#plt.plot(x, gaus2(x, 1500))
#plt.plot(x, 1/np.sqrt(x/200))
plt.show()


