#!/usr/bin/env python3

import cloudpickle as pkl
import numpy as np

from mfec.klt import KLT


class MinMaxRecorder:
    def __init__(self, dim):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        self.maxes = -np.ones(dim) * np.inf
        self.mins = np.ones(dim) * np.inf
        self.ndimensions = dim

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        data = np.atleast_2d(data)
        if data.shape[1] != self.ndimensions:
            raise ValueError("Data dims don't match prev observations.")
        new_mins = np.min(data, axis=0)
        new_maxes = np.max(data, axis=0)

        self.maxes = np.maximum(new_maxes, self.maxes)
        self.mins = np.minimum(new_mins, self.mins)


class StatsRecorder:
    """modified from https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    This definietly works"""

    def __init__(self, dim):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        self.mean = np.zeros(dim)
        self.std = np.ones(dim)
        self.nobservations = 0
        self.ndimensions = dim

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        data = np.atleast_2d(data)
        if data.shape[1] != self.ndimensions:
            raise ValueError("Data dims don't match prev observations.")

        newmean = data.mean(axis=0)
        newstd = data.std(axis=0)

        m = self.nobservations * 1.0
        n = data.shape[0]

        tmp = self.mean

        self.mean = m / (m + n) * tmp + n / (m + n) * newmean
        self.std = m / (m + n) * self.std ** 2 + n / (m + n) * newstd ** 2 + \
                   m * n / (m + n) ** 2 * (tmp - newmean) ** 2
        self.std = np.sqrt(self.std)

        self.nobservations += n


class MFECAgent:
    def __init__(
            self,
            buffer_size,
            k,
            discount,
            epsilon,
            observation_dim,
            state_dimension,
            actions,
            seed,
            epsilon_decay,
            clip_rewards,
            M,
            norm_freq,
    ):
        self.rs = np.random.RandomState(seed)
        self.actions = actions
        self.klt = KLT(actions=self.actions,
                       buffer_size=buffer_size,
                       k=k,
                       state_dim=state_dimension,
                       M=M,
                       seed=seed)

        self.discount = discount
        self.norm_freq = norm_freq
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action = int
        self.train_count = 0

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

    def choose_action(self, observation, time):
        # state = self.transformer.transform(observation.reshape(1, -1))[0]
        state = observation.flatten()

        query_results = np.asarray([
            self.klt.estimate(state, action, time)
            for action in self.actions])

        r_estimates = query_results[:, 0]
        d_estimates = query_results[:, 1]

        print(d_estimates)
        r_estimates = r_estimates / (d_estimates+0.001)
        # Exploration
        if self.rs.random_sample() < self.epsilon:
            action = np.random.choice(self.actions)
        # Exploitation
        else:
            probs = np.zeros_like(self.actions)
            probs[np.where(r_estimates == max(r_estimates))] = 1
            probs = probs / sum(probs)

            action = self.rs.choice(self.actions, p=probs)

        bonus = 0 #d_estimates[action]
        return action, state, bonus, 0

    def train(self, trace):
        R = 0.0
        for i in range(len(trace)):
            experience = trace.pop()
            s = experience["state"]
            a = experience["action"]
            t = experience["time"]
            b = experience["bonus"]

            r = self.clipper(experience["reward"]) + b

            if i == 0:
                # last sample
                R = r
            else:
                R = r + self.discount * R

            self.klt.update(s, a, R, t)

        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(f"eps={self.epsilon:.2f}")

    def save(self, save_dir):
        with open(f"{save_dir}/agent.pkl", "wb") as f:
            pkl.dump(self, f)
