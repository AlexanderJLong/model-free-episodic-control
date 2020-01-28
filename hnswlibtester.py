import hnswlib
import numpy as np
import pickle as pkl
import random
from tqdm import tqdm
dim = 400
k=160
num_elements = 50_000

# Generating sample data
data = pkl.load(open("saves/states.pkl", "rb"))

print(np.asarray(data).shape)

p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip
# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 20)
p.add_items(data)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(data)

recalls = []
for _ in tqdm(range(1000)):
    sample = data[random.randint(0, num_elements)]
    approx, _ = p.knn_query(sample, k = k)
    _, exact = nbrs.kneighbors([sample])

    approx = approx[0]
    exact = exact[0]

    recall = sum([a in exact for a in approx])/k
    recalls.append(recall)
    print(recall)

print(np.mean(recalls))



"""get exact neighbours"""

distances, indices = nbrs.kneighbors([data[5143]])

print(indices, distances)
