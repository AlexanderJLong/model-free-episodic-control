import hnswlib
import numpy as np

dim = 128
num_elements = 4

# Generating sample data
data = np.asarray([
    [1, 1, 3],
    [1, 1, 1],
    [1, 1, 3],
    [1, 1, 2],]

)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = 3) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = 100, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
p.add_items(data)


# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query([1, 1, 3], k = 4)
print(labels)
print(distances)