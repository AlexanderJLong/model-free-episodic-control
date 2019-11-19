#!/usr/bin/env python3

import os.path
import pickle

import numpy as np

from mfec.qec import QEC


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
            exp_skip,
            autonormalization_frequency,
            epsilon_decay,
            kernel_type,
            kernel_width,
            projection_type,
    ):
        self.rs = np.random.RandomState(seed)
        self.memory = []
        self.actions = actions
        self.qec = QEC(self.actions, buffer_size, k, kernel_type, kernel_width, state_dimension, seed)

        self.training = True  # set to false to act greedily

        if projection_type == 0:
            self.projection = np.eye(state_dimension)[:, :observation_dim]
        elif projection_type == 1:
            self.projection = self.rs.randn(
                state_dimension, observation_dim
            ).astype(np.float32)
        elif projection_type == 2:
            self.projection = np.linalg.qr(self.rs.randn(
                state_dimension, observation_dim
            ).astype(np.float32))[0]
        elif projection_type == 3:
            m = []
            for i in range(state_dimension):
                r = []
                for j in range(observation_dim):
                    d = np.random.rand()
                    if d < 1 / 6:
                        r.append(1)
                    elif d < 5 / 6:
                        r.append(0)
                    else:
                        r.append(-1)
                m.append(r)
            self.projection = np.asarray(m, dtype=np.int16)

        print(self.projection.shape)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.autonormalization_frequency = autonormalization_frequency
        self.state = np.empty(state_dimension, self.projection.dtype)
        self.action = int
        self.time = 0
        self.rewards_received = 0
        self.exp_skip = exp_skip
        self.t = 0  # keep track of episode step

    def choose_action(self, observation):
        self.time += 1

        # Preprocess and project observation to state
        self.state = np.dot(self.projection, np.asarray(observation).flatten())
        # self.state = observation

        # Exploration
        if self.rs.random_sample() < self.epsilon and self.training:
            self.action = self.rs.choice(self.actions)

        # Exploitation
        else:
            values = [self.qec.estimate(self.state, action) for action in self.actions]
            best_actions = np.argwhere(values == np.max(values)).flatten()
            self.action = self.rs.choice(best_actions)

        return self.action, self.state

    def train(self, trace):
        # Takes trace object: a list of dicts {"state", "action", "reward"}
        value = 0.0
        for _ in range(len(trace)):
            experience = trace.pop()
            value = value * self.discount + experience["reward"]
            self.qec.update(
                experience["state"],
                experience["action"],
                value,
            )

        # Decay e exponentially
        if self.epsilon > 0:
            self.epsilon /= 1 + self.epsilon_decay
            print(self.epsilon)

    def save(self, results_dir):
        with open(os.path.join(results_dir, "agent.pkl"), "wb") as file:
            pickle.dump(self, file, 2)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
