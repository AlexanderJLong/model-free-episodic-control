#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors.kd_tree import KDTree
import matplotlib.pyplot as plt


class QEC:
    def __init__(self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k

    def estimate(self, state, action):
        buffer = self.buffers[action]
        if len(buffer) <= self.k:
            return float("inf")

        value = 0.0
        neighbors = buffer.find_neighbors(state, self.k)
        if np.allclose(buffer.states[neighbors[0]], state):
            return buffer.values[neighbors[0]]
        else:
            for neighbor in neighbors:
                value += buffer.values[neighbor]
            return value / self.k

    def update(self, state, action, value, time, step):
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time, step)

    def plot(self):
        if len(self.buffers[0].states) < 2:
            return
        fig, axes = plt.subplots(4,3)
        for i in range(2):
            buffer = self.buffers[i]
            for j in range(4):
                states = np.asarray(buffer.states)
                vals = np.asarray(buffer.values)
                steps = np.asarray(buffer.steps)
                axes[j,i].scatter(states[::2, j], vals[::2], c=steps[::2], alpha=0.5)
        # Run a regressor over 1st dim grid and then also show the dif between the two
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

    def find_neighbors(self, state, k):
        return self._tree.query([state], k)[1][0] if self._tree else []

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


