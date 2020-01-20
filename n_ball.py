import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma


def nball(r, n):
    R = np.power(r, n, dtype=np.float128)
    V = np.power(np.pi, n / 2) / gamma(n / 2 + 1)
    return R*V



def density(d, n=50_000):
    return np.sqrt(n)/(n * nball(d, 16))


d = np.linspace(1, 2, 10000)

plt.plot(d, density(d))
plt.show()
