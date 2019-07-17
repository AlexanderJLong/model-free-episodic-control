#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors.kd_tree import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class QEC:
    def __init__(self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k

    def estimate(self, state, action):
        buffer = self.buffers[action]
        if len(buffer) <= self.k:
            return float("inf")

        neighbors, dists = buffer.find_neighbors(state, self.k, ball=False)

        dists = dists[0]
        neighbors = neighbors[0]
        if len(neighbors) == 0:
            return float("inf")

        # if np.allclose(buffer.states[neighbors[0]], state):
        #    print("Exact Match")
        #    return buffer.values[neighbors[0]]
        else:
            w = [1 for d in dists]
            value = 0
            for i, neighbor in enumerate(neighbors):
                value += w[i] * buffer.values[neighbor]

            if sum(w) == 0:
                return float("inf")
            return value / sum(w)

    def update(self, state, action, value, time, step):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time, step)

    def plot(self, skip_factor):
        if len(self.buffers[0].states) < 2:
            return
        fig, axes = plt.subplots(4, 3)
        for j in range(4):
            Ks = []
            for i in range(2):
                buffer = self.buffers[i]
                states = np.asarray(buffer.states)
                vals = np.asarray(buffer.values)
                steps = np.asarray(buffer.steps)
                axes[j, i].scatter(states[::skip_factor, j], vals[::skip_factor], c=steps[::skip_factor], alpha=0.5)

                # Do the interpolation
                T = np.linspace(-2.5, 2.5, 4000)
                k = []
                for t in T:
                    state = [0, 0, 0, 0]
                    state[j] = t
                    k.append(self.estimate(state, i))
                Ks.append(np.asarray(k))
                axes[j, i].plot(T, k, c="r")
            axes[j, 2].plot(T, Ks[0] - Ks[1])

        # Run a regressor over 1st dim grid and then also show the dif between the two
        axes[0, 0].set(title='Q(a0)')
        axes[0, 1].set(title='Q(a1)')
        axes[0, 2].set(title='Advantage(a0-a1)')

        axes[0, 0].set(ylabel='Position')
        axes[1, 0].set(ylabel='Velocity')
        axes[2, 0].set(ylabel='Angle')
        axes[3, 0].set(ylabel='Vel. at tip.')
        plt.show()

    def plot3d(self):
        """start with just position and angle"""
        fig = plt.figure()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111, projection='3d')
        data = self.buffers[0]
        states = np.asarray(data.states)
        vals = np.asarray(data.values)
        ax.scatter(states[:, 0], states[:, 1], states[:,2], c=vals)
        ax.set(xlabel="Position")
        ax.set(ylabel="Vel")
        ax.set(zlabel="Angle")
        plt.show()


class ActionBuffer:
    def __init__(self, capacity):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []
        self.steps = []

    def find_state(self, state):
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k, ball):
        """Return idx, dists"""
        if ball:
            return self._tree.query_radius([state], r=0.3, return_distance=True) if self._tree else []
        else:
            result = self._tree.query([state], k=k, return_distance=True) if self._tree else []
            return result[1], result[0]

    def add(self, state, value, time, step):
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
            self.steps.append(step)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(np.asarray(self.states))

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)
