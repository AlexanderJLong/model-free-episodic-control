#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors.kd_tree import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class QEC:
    def __init__(self, actions, buffer_size, k, kernel_width, state_dim):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])
        self.k = k
        self.mu = np.zeros(state_dim)  # offset
        self.sig = np.ones(state_dim)  # scale
        self.kernel_width = kernel_width

    def get_mu_and_sig(self):
        """get the average mean and std deviation of each dim over all buffers"""
        mus = []
        sigs = []
        for buff in self.buffers:
            mus.append(np.mean(buff.states, axis=0))
            sigs.append(np.std(buff.states, axis=0))
        return np.mean(mus, axis=0), np.mean(sigs, axis=0)

    def autonormalize(self):
        """NOTE: NO MU - won't work if state vars arent centered
        should really check all buffer sizes as well, but this might be slow"""
        if len(self.buffers[0]) <= self.k:
            return
        """change all states, in all buffers, to refect the changes in scaling factors"""
        mu, sig = self.get_mu_and_sig()
        sig[sig == 0] = 1
        for buff in self.buffers:
            buff.states = ((buff.states - mu) / sig).tolist()
        self.mu = self.mu + self.mu / self.sig
        self.sig = sig * self.sig
        #print(self.get_mu_and_sig())
        #print(self.mu, self.sig)
        return

    def estimate(self, state, action, step):
        state = (state - self.mu) / self.sig
        """Changes:
        - No exact matching"""
        buffer = self.buffers[action]
        if len(buffer) < self.k:
            return float("inf")

        neighbors, dists = buffer.find_neighbors(state, self.k, ball=False)
        dists = dists[0]
        neighbors = neighbors[0]

        def gaus(x, sig):
            return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power(x / sig, 2.) / 2)

        w = gaus(dists, self.kernel_width)
        assert (len(w) == len(dists))
        value = 0
        # a = 0
        for i, neighbor in enumerate(neighbors):
            # only look at states forward in time
            # print(f"{step} -> {buffer.steps[neighbor]}")
            # amount_ahead = buffer.steps[neighbor] - step
            # if amount_ahead >= 0:
            #     a += 1
            value += w[i] * buffer.values[neighbor]
        # print(f"% ahead in NN search: {a/len(neighbors)*100}")
        if sum(w) == 0:
            return 0
        # print(value)
        return value / sum(w)

    def update(self, state, action, value, time, step):
        #NOTE: in replace, the step will be replaced with the latest
        # This might not be desired behaviour
        state = (state - self.mu) / self.sig
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index, step)
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
                states = np.asarray(data.states)
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
            ax1 = fig.add_subplot(111, projection='3d')
            fig.set_tight_layout(True)
            maps= ["Blues", "Reds"]
            for i in range(2):
                data = self.buffers[i]
                states = np.asarray(data.states)
                vals = np.asarray(data.values)
                ax1.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals, cmap=maps[i])

            ax1.set(xlabel="Vel")
            ax1.set(ylabel="Angle")
            ax1.set(zlabel="Position")
        plt.show()

    def plot_scatter(self):

        def turn_on_grid(ax):
            ax.grid(b=True, which='major', color='#666666', linestyle='-')
            # Show the minor grid lines with very faint and almost transparent grey lines
            ax.minorticks_on()
            ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        fig = plt.figure()
        fig.set_tight_layout(True)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        maps = ["Reds", "Blues"]
        for i in range(2):
            data = self.buffers[i]
            states = np.asarray(data.states)
            vals = np.asarray(data.values)
            im1 = ax1.scatter(states[:, 1], states[:, 2], c=vals, cmap=maps[i])

        states = np.random.rand(5000, 4) * 8 - 4
        states_to_feed = states * self.sig + self.mu
        states_to_feed[:, 3] = 0
        states_to_feed[:, 0] = 0
        e0 = []
        e1 = []
        for s in states_to_feed:
            e0.append(self.estimate(s, 0, 0))
            e1.append(self.estimate(s, 1, 0))
        diff = np.asarray(e0) - np.asarray(e1)
        # force normalization between certain range and make sure its symetric
        diff[0] = max(max(diff), -min(diff))
        diff[1] = min(-max(diff), min(diff))
        im2 = ax2.scatter(states[:, 1], states[:, 2], c=diff, cmap="bwr")
        im3 = ax3.scatter(states[:, 1], states[:, 2], c=e0, cmap="Reds")
        im4 = ax4.scatter(states[:, 1], states[:, 2], c=e1, cmap="Blues")

        for im, ax in [(im1, ax1), (im2, ax2), (im3, ax3), (im4, ax4)]:
            turn_on_grid(ax)
            ax.set(xlabel="Vel")
            ax.set(ylim=[-4, 4])
            ax.set(xlim=[-4, 4])
            ax.set(ylabel="Angle")
            fig.colorbar(im, ax=ax)
            ax.set(title=f"max={max(vals):.2f}, min={min(vals):.2f}")
        plt.show()
        return


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

    def replace(self, state, value, time, index, step):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time
        self.steps[index] = step

    def __len__(self):
        return len(self.states)
