#!/usr/bin/env python3

import os.path
import pickle

import numpy as np
from sklearn import random_projection

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
            epsilon_decay,
            warmup,
            distance,
    ):
        self.rs = np.random.RandomState(seed)
        self.actions = actions
        self.qec = QEC(self.actions, buffer_size, k, state_dimension, distance, warmup, seed)

        self.transformer = random_projection.SparseRandomProjection(n_components=state_dimension, dense_output=True)
        self.transformer.fit(np.zeros([1, observation_dim]))
        # self.transformer.components_.data[np.where(self.transformer.components_.data < 0)] = -1
        # self.transformer.components_.data[np.where(self.transformer.components_.data > 0)] = 1
        # self.transformer.components_ = self.transformer.components_.astype(np.int8)

        for r in self.transformer.components_:
            print(r)

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action = int
        self.training = False

    def choose_action(self, observation):

        # Preprocess and project observation to state
        # print(observation)
        self.state = self.transformer.transform(observation.reshape(1, -1))
        # self.state = observation.reshape(1, -1)

        # print(self.state)
        # print(self.state)
        # self.state = observation.flatten()
        # print(self.transformer.components_.dtype)
        # self.state = np.asarray(self.state, dtype=np.int16)
        # self.state = self.projection @ observation.flatten()
        # print(self.state.dtype)
        # self.state = observation

        # Exploration
        # if self.rs.random_sample() < self.epsilon and self.training:
        #    self.action = self.rs.choice(self.actions)

        # Exploitation
        # else:
        values = np.asarray(
            [self.qec.estimate(self.state, action, use_count_exploration=self.training)
             for action in self.actions
             ])
        maxes = np.where(values == max(values))
        probs = np.zeros_like(self.actions)
        probs[maxes] = 1
        # values = np.power(values+1, 5)
        probs = probs / sum(probs)
        # best_actions = np.argwhere(values == np.max(values)).flatten()
        self.action = self.rs.choice(self.actions, p=probs)

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

        self.qec.solidify_values()
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
