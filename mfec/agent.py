#!/usr/bin/env python3

import os.path
import pickle

import numpy as np
from scipy.misc.pilutil import imresize

from mfec.qec import QEC


class MFECAgent:
    def __init__(
            self,
            buffer_size,
            k,
            discount,
            epsilon,
            height,
            width,
            state_dimension,
            actions,
            seed,
            exp_skip,
            autonormalization_frequency,
            epsilon_decay,
    ):
        self.rs = np.random.RandomState(seed)
        self.size = (height, width)
        self.memory = []
        self.actions = actions
        self.qec = QEC(self.actions, buffer_size, k)
        self.projection = self.rs.randn(
            state_dimension, height * width
        ).astype(np.float32)
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

    def choose_action(self, observation, step):
        self.time += 1

        # Preprocess and project observation to state
        # self.state = np.dot(self.projection, np.asarray(observation).flatten())
        self.state = observation

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            self.action = self.rs.choice(self.actions)

        # Exploitation
        else:
            values = [
                self.qec.estimate(self.state, action, step)
                for action in self.actions
            ]
            best_actions = np.argwhere(values == np.max(values)).flatten()
            self.action = self.rs.choice(best_actions)
            # print(f"In {observation}, got values {values} and picked {self.action}")

        return self.action

    def receive_reward(self, reward, step):
        self.memory.append(
            {
                "state": self.state,
                "action": self.action,
                "reward": reward,
                "time": self.time,
                "step": step,
            }
        )

    def train(self):
        self.rewards_received += 1
        value = 0.0
        for _ in range(len(self.memory)):
            experience = self.memory.pop()
            value = value * self.discount + experience["reward"]
            if self.rewards_received % self.exp_skip == 0:
                self.qec.update(
                    experience["state"],
                    experience["action"],
                    value,
                    experience["time"],
                    experience["step"],
                )

        #Normalize
        if self.autonormalization_frequency is not 0:
            if not self.rewards_received % self.autonormalization_frequency:
                self.qec.autonormalize()

        # Decay e linearly
        if self.epsilon > 0:
            self.epsilon -= self.epsilon_decay

    def save(self, results_dir):
        with open(os.path.join(results_dir, "agent.pkl"), "wb") as file:
            pickle.dump(self, file, 2)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
