# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 1000, 100)
Y = np.arange(0, 1000, 100)
X, Y = np.meshgrid(X, Y)


def gaus_2d(x, y, sig1, sig2):
    return np.exp(-(np.square(x / sig1) + np.square(y / sig2)))


def laplace_2d( x, y, sig1, sig2):
    return np.exp(-(x / sig1) - (y / sig2))

Z = laplace_2d(X, Y, 300, 1000)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.set_xlabel("dist")
ax.set_ylabel("time")
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()