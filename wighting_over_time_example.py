import numpy as np
import matplotlib.pyplot as plt
initial_dist = np.random.uniform(0, 10_000, 2000)


def gaus2(x, sig):
    return np.exp(-np.square(x / sig))
plt.plot(initial_dist, np.zeros_like(initial_dist))
plt.show()


print(initial_dist)
