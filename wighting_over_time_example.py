import hnswlib
import numpy as np
import time
dim = 128
num_elements = 100_000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 10, M = 16)

t0 = time.time()
# Element insertion (can be called several times):
p.add_items(data)
t1 = time.time()

print(t1-t0)