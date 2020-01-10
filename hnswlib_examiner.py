import hnswlib
import numpy as np

dim = 128
num_elements = 4

# Generating sample data
data = np.asarray([
    [1, 1, 3],
    [1, 1, 1],
    [1, 1, 3],
    [1, 1, 2],
    [1, 1, 3],
]
)

# Declaring index
p = hnswlib.Index(space='l2', dim=3)  # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=100, ef_construction=200, M=16)

# Element insertion (can be called several times):
print(p.get_ids_list())

p.add_items(data)
print(p.get_ids_list())

print(p.get_items(np.asarray(p.get_ids_list())))
#p.mark_deleted(0)
#p.mark_deleted(4)
print(p.get_items([1, 1, 3]))


# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query([1, 1, 3], k=10)
print(labels)
print(distances)
