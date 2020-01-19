import hnswlib
import numpy as np


def gaus(x, sig):
    return np.exp(-np.square(x / sig) / 2)


tree = hnswlib.Index(space="l2", dim=1)  # possible options are l2, cosine or ip
tree.init_index(max_elements=100_001, M=30)
tree.add_items([5])
samples = []
x = np.linspace(-10, 10, 500)

for n in range(100_000):
    point = np.random.normal(3, 2)
    tree.add_items([point])

    # Get estimate
    if n and not n % 1000:
        k = int(np.sqrt(n))
        print(x)
        _, dists = tree.knn_query(x, k=k)
        print(dists.shape)
        density = (k/n)/np.max(dists, axis=0)
        #print(len(density))
