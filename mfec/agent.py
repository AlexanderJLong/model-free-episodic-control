#!/usr/bin/env python3

import os.path
import pickle
from PIL import Image
import numpy as np

from mfec.qec import QEC


# Different Preprocessors
def cartpole_crop_grey_scale_normalize_resize(obv):
    # Greyscale and normalize to [0, 1]
    state = obv[:, :, 0] * 0.001172549019607843 + obv[:, :, 1] * 0.0023019607843137255 + obv[:, :, 2] * 0.0004470588235294118

    state = state - 0.5
    # resize
    state = np.array(Image.fromarray(state).resize((84, 84), Image.BILINEAR), dtype=np.float32)

    # round
    #state = np.around(state, decimals=2)
    #print(np.max(state), np.min(state))

    return state

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
        self.qec = QEC(self.actions, buffer_size, k, kernel_type, kernel_width, state_dimension)

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.state = np.empty(state_dimension)
        self.action = 0

    def choose_action(self, observation):
        values = [self.qec.estimate(observation, action) for action in self.actions]
        best_actions = np.argwhere(values == np.max(values)).flatten()
        action = self.rs.choice(best_actions)
        return action, values

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

        # Decay e linearly
        if self.epsilon > 0:
            self.epsilon -= self.epsilon_decay
