#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, obv_dim, distance, lr, seed):
        self.buffers = tuple([ActionBuffer(buffer_size, state_dim, distance, lr,  seed) for _ in actions])
        self.k = k
        self.obv_dim = obv_dim  # dimentionality of origional data

    def estimate(self, state, action, count_weight, training):
        """Return the estimated value of the given state"""

        buffer = self.buffers[action]

        if len(buffer) == 0:
            return 0

        k = min(self.k, len(buffer))  # the len call might slow it down a bit
        neighbors, dists = buffer.find_neighbors(state, k)
        # Strip batch dim. Note dists is already ordered.
        dists = dists[0]
        neighbors = neighbors[0]

        # print(dists, neighbors, buffer.values_array, action)
        if dists[0] == 0:
            # Identical state found
            weighted_reward = buffer.values_array[neighbors[0]]
            #weighted_count = 1./np.sqrt(buffer.counts_array[neighbors[0]])
            # w = [0]
        else:

            # never seen before so estimate
            values = buffer.values_array[neighbors]
            #counts = buffer.counts_array[neighbors]

            # Convert to l2norm, normalize by original dimensionality so dists have a consistent
            # range, but make sure they're always still > 1 because of w=1/d
            norms = np.sqrt(dists / self.obv_dim)
            #norms = dists / self.obv_dim

            # return sum(buffer.values[n] for n in neighbors)
            w = np.divide(1., norms)  # Get inverse distances as weights
            # dist_weighted_count = np.sum(w * buffer.counts_array[neighbors]) / np.sum(w)
            weighted_reward = np.sum(w * values) / np.sum(w)
            #print(weighted_reward)
            # weighted_reward = np.sum(w * values)
            #weighted_count = np.sum(w * np.power(counts, -0.5)) / np.sum(w)
            #print(counts, weighted_count)

        #print(f"r:{weighted_reward} + r'{weighted_count * training}, rd:{np.mean(w)}")

        return weighted_reward #+ count_weight * ( weighted_count + np.mean(w)) * training # No exploration bonus on testing

    def update(self, state, action, value):
        # print("updating", action)
        buffer = self.buffers[action]
        buffer.add(state, value)

    def solidify_values(self):
        for b in self.buffers:
            b.solidify_values()

    def plot3d(self, both, diff):
        fig = plt.figure()
        fig.set_tight_layout(True)
        if diff and both:
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')
            axes = [ax1, ax2]
            for i, ax in enumerate(axes):
                data = self.buffers[i]
                states = np.asarray(data.get_states())
                vals = np.asarray(data.values)
                ax.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals)
                ax.set(xlabel="Vel")
                ax.set(ylabel="Angle")
                ax.set(zlabel="Position")

                ax.set(title=f"max r={max(vals)}")

            states = np.random.rand(5000, 4) * 5 - 2
            states[:, -1] = 0
            vals = []
            for s in states:
                vals.append(self.estimate(s, 1, 0) - self.estimate(s, 0, 0))

            # force normalization between certain range and make sure its symetric
            vals[0] = max(max(vals), -min(vals))
            vals[1] = min(-max(vals), min(vals))
            ax3.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals, cmap="bwr")
            ax3.set(xlabel="Vel")
            ax3.set(ylabel="Angle")
            ax3.set(zlabel="Position")
            ax3.set(title=f"max={max(vals):.2f}, min={min(vals):.2f}")
            plt.show()
            return

        elif diff:
            ax = fig.add_subplot(111, projection='3d')
            states = np.random.rand(5000, 4) * 5 - 2
            vals = []
            for s in states:
                vals.append(self.estimate(s, 1, 0))
            ax.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals)

            ax.set(xlabel="Vel")
            ax.set(ylabel="Angle")
            ax.set(zlabel="Position")
            plt.show()
            return

        else:
            if len(self.buffers[0].values_list) < 10:
                return
            num_actions = len(self.buffers)
            cols = 4
            rows = num_actions // cols + 1
            max_r = max([max(b.values_list) for b in self.buffers])
            for i in range(num_actions):
                ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

                data = self.buffers[i]
                states = np.asarray(data.get_states())[::]
                if len(states) < 1:
                    return
                vals = np.asarray(data.values)[::]
                ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=vals, vmax=max_r)

            plt.show()
        return


class ActionBuffer:
    def __init__(self, capacity, state_dim, distance, lr, seed):
        self.state_dim = state_dim
        self.lr = lr
        self.capacity = capacity
        self._tree = hnswlib.Index(space=distance, dim=state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity, M=20, random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.
        self.values_array = np.asarray([])  # For lookup. Update at train by converting values_list.
        self.counts_list = []
        self.counts_array = np.asarray([])

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(state, k=k)

    def add(self, state, value):
        if not self.values_list:  # buffer empty, just add
            self._tree.add_items(state)
            self.values_list.append(value)
            self.counts_list.append(1)
            return

        idx, dist = self.find_neighbors(state, 1)
        idx = idx[0][0]
        dist = dist[0][0]
        if dist < 1e-6 or np.isnan(dist):
            # Existing state, update and return
            self.counts_list[idx] += 1
            self.values_list[idx] = (1 - 1/self.counts_list[idx])*self.values_list[idx] + (1/self.counts_list[idx]) * value
            #self.values_list[idx] = 0.7*self.values_list[idx] +0.3*value
            #print(f"updating {self.values_list[idx]}")
            #self.values_list[idx] = (1-self.lr)*self.values_list[idx] * self.lr*value

        else:
            #print(f"adding {value}")
            self.values_list.append(value)
            self._tree.add_items(state)
            self.counts_list.append(1)

        return

    def solidify_values(self):
        self.values_array = np.asarray(self.values_list)
        self.counts_array = np.asarray(self.counts_list)

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        return len(self.values_array)
