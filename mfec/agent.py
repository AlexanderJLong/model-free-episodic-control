#!/usr/bin/env python3

import os.path
import pickle
from PIL import Image
import numpy as np

from mfec.qec import QEC


# Different Preprocessors
def cartpole_crop_grey_scale_normalize_resize(obvs):
    states = []
    for obv in obvs:
        state = obv[:, :, 0] * 0.001172549019607843 + obv[:, :, 1] * 0.0023019607843137255 + obv[:, :, 2] * 0.0004470588235294118

        # resize
        state = np.array(Image.fromarray(state).resize((64, 64)))
        states.append(state.flatten())
    return np.asarray(states)


class MFECAgent:
    def __init__(
            self,
            num_envs,
            buffer_size,
            k,
            discount,
            prepro,
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
        self.memory = [[] for _ in range(num_envs)]
        self.actions = actions
        self.qec = QEC(self.actions, buffer_size, k, kernel_type, kernel_width, state_dimension)

        if prepro == "GreyScaleNormalizeResize":
            self.prepro = cartpole_crop_grey_scale_normalize_resize
            obv_dim = 64 * 64
        else:
            obv_dim = observation_dim

        if projection_type == 0:
            self.projection = np.eye(state_dimension)[:, :obv_dim]
        elif projection_type == 1:
            self.projection = self.rs.randn(
                state_dimension, obv_dim
            ).astype(np.float32)
        elif projection_type == 2:
            self.projection = np.linalg.qr(self.rs.randn(
                state_dimension, obv_dim
            ).astype(np.float32))[0]
        elif projection_type == 3:
            m = []
            for i in range(state_dimension):
                r = []
                for j in range(obv_dim):
                    d = np.random.rand()
                    if d < 1 / 6:
                        r.append(1)
                    elif d < 5 / 6:
                        r.append(0)
                    else:
                        r.append(-1)
                m.append(r)
            self.projection = np.asarray(m)
        elif projection_type == 4:
            self.projection = np.asarray([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 10, 0],
                [0, 0, 0, 1]])
        elif projection_type == 5:
            self.projection = np.asarray([
                [0.5, 0.5, 0, 0],
                [0, 0.5, 0.5, 0],
                [0, 0, 0.5, 0.5],
                [0.5, 0, 0, 0.5]])
        elif projection_type == 6:
            self.projection = np.asarray([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 10, 0],
                [0, 0, 0, 1]])

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.autonormalization_frequency = autonormalization_frequency
        self.state = np.empty(state_dimension, self.projection.dtype)
        self.action = int
        self.time = 0
        self.rewards_received = 0
        self.exp_skip = exp_skip

    def choose_action(self, observations):
        self.time += 1

        # Preprocess and project observation to state
        obvs = self.prepro(observations)
        self.states = np.dot(self.projection, obvs)

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            self.action = self.rs.choice(self.actions)

        # Exploitation
        else:
            values = [
                self.qec.estimate(self.state, action)
                for action in self.actions
            ]
            best_actions = np.argwhere(values == np.max(values)).flatten()
            self.action = self.rs.choice(best_actions)
            # print(f"In {observation}, got values {values} and picked {self.action}")

        return self.action

    def store_rewards(self, rewards, dones):
        """batch setting, have list of everything, one for each env.
        Want to append these separately"""

        for i, env_mem in enumerate(self.memory):
            env_mem.append(
                {
                    "state": self.states[i],
                    "action": self.actions[i],
                    "reward": rewards[i],
                    "done": dones[i],
                    "time": self.time,
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
                )

        # Normalize
        if self.autonormalization_frequency is not 0:
            if not self.rewards_received % self.autonormalization_frequency:
                self.qec.autonormalize()

        # Decay e linearly
        if self.epsilon > 0:
            self.epsilon -= self.epsilon_decay
