#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import umap


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, obv_dim, distance, seed):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.buffers = tuple(
            [ActionBuffer(a, self.buffer_size, state_dim, distance, seed) for a in actions])
        self.k = k
        self.obv_dim = obv_dim  # dimentionality of origional data

    def gaus(self, x, sig):
        return np.exp(-np.square(x/sig)/2)

    def estimate(self, state, action, time):
        """
        Return:
         1) the estimated value of the given state
         2) a measure of distance to surrounding samples

         According to this measure we don't care once we have more than k samples on a point, it just gets treated
         the same as we add more. So need a reasonably high k.
         """

        buffer = self.buffers[action]
        if buffer.length == 0:
            return 0, 0  # Maybe better to just signal the buffer is empty than assigning large dist

        k = min(buffer.length, self.k)
        neighbors, dists = buffer.find_neighbors(state, k)
        # Strip batch dim. dists is already ordered. Note it is square of l2 norm.
        neighbors = neighbors[0]
        dists = np.sqrt(dists[0])

        values = [buffer.values_list[n] for n in neighbors]
        times = [buffer.time_list[n] for n in neighbors]

        z = 1/np.sqrt((time-np.asarray(times)))

        if dists[0] == 0:
            time_weighted = values[0]
        else:
            w = 1/np.sqrt(dists)
            #print(w, z)
            w = w/np.max(w)
            z = z/np.max(z)
            c = 1*w + 0.0*z
            time_weighted = np.dot(values, c) / np.sum(c)
        #print(time_weighted)
        #print(np.mean(values))

        return time_weighted, [0,0]

    def update(self, state, action, value, time):
        # print("updating", action)
        buffer = self.buffers[action]
        buffer.add(state, value, time)


    def save_indexes(self, save_dir):
        """
        Serialize the index and values. Must be done this way because hnswlib index's cannot be pickled
        """
        for i, buff in enumerate(self.buffers):
            buff._tree.save_index(f"{save_dir}/buff_{i}.bin")

    def load_indexes(self, save_dir):
        """
        Load the index and values. Assumes index has already been initialized
        """
        for i, buff in enumerate(self.buffers):
            buff._tree.load_index(f"{save_dir}/buff_{i}.bin", max_elements=self.buffer_size)

    def plot3d(self,):
        fig = plt.figure()
        reducer = umap.UMAP(n_neighbors=200, n_components=2)

        fig.set_tight_layout(True)
        rows = 4
        for i, buffer in enumerate(self.buffers):
            ax = fig.add_subplot(rows, len(self.buffers)//rows+1, i+1)

            states = np.asarray(buffer.get_states())
            embeddings = reducer.fit_transform(states)
            vals = np.asarray(buffer.values_list)
            ax.scatter(embeddings[:, 1], embeddings[:, 0], c=vals, s=1)
            ax.set(xlabel="Vel")
            ax.set(ylabel="Angle")

            ax.set(title=f"max r={max(vals)}")
        plt.show()
        return



class ActionBuffer:
    def __init__(self, n, capacity, state_dim, distance, seed):
        self.id = n
        self.state_dim = state_dim
        self.capacity = capacity
        self.distance = distance
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity, M=100, random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.
        self.time_list = []
        self.length = 0
        self.agg_dist = 1e6

    def __getstate__(self):
        # pickle everything but the hnswlib indexes
        self._tree.save_index(f"saves/index_{self.id}.bin")
        return dict((k, v) for (k, v) in self.__dict__.items() if k != "_tree")

    def __setstate__(self, d):
        self.__dict__ = d
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)
        self._tree.load_index(f"saves/index_{self.id}.bin")

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(state, k=k)

    def add(self, state, value, time):
        if not self.values_list:  # buffer empty, just add
            self._tree.add_items(state)
            self.values_list.append(value)
            self.time_list.append(time)
            return

        idx, dist = self.find_neighbors(state, 1)
        idx = idx[0][0]
        dist = dist[0][0]
        if dist < self.agg_dist or dist < 1:
            # Existing state, update and return
            self.values_list[idx] = 0.5*value + 0.5*self.values_list[idx]
            self.time_list[idx] = time
        else:
            self.values_list.append(value)
            self.time_list.append(time)
            self._tree.add_items(state)

        self.length = len(self.values_list)

    def remove(self, idx):
        """Remove a sample"""
        self._tree.mark_deleted(idx)

    def get_states(self):
        return self._tree.get_items(range(0, self.length))

