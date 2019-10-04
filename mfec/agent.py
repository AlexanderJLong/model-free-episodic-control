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

def pong_prepro(obv):
    # Crop, Greyscale and normalize
    state = obv[35:190, :, 0] * 0.001172549019607843 + obv[35:190, :, 1] * 0.0023019607843137255 + obv[35:190, :, 2] * 0.0004470588235294118

    # resize
    state = np.array(Image.fromarray(state).resize((64, 64), Image.BILINEAR), dtype=np.float32)

    # round
    #state = np.around(state, decimals=1)

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

        if observation_dim > 4:
            self.prepro = cartpole_crop_grey_scale_normalize_resize
        else:
            self.prepro = lambda x:x


        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.autonormalization_frequency = autonormalization_frequency
        self.state = np.empty(state_dimension)
        self.action = int
        self.time = 0
        self.rewards_received = 0
        self.exp_skip = exp_skip

    def choose_action(self, observation):
        # Preprocess and project observation to state
        state = self.prepro(observation)
        #self.state = np.dot(self.projection, state.flatten())
        self.state = state.flatten()

        values = [
            self.qec.estimate(self.state, action)
            for action in self.actions
        ]
        best_actions = np.argwhere(values == np.max(values)).flatten()
        self.action = self.rs.choice(best_actions)

        return self.action, values

    def receive_reward(self, reward):
        self.memory.append(
            {
                "state": self.state,
                "action": self.action,
                "reward": reward,
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
                )

        # Normalize
        if self.autonormalization_frequency is not 0:
            if not self.rewards_received % self.autonormalization_frequency:
                self.qec.autonormalize()

        # Decay e linearly
        if self.epsilon > 0:
            self.epsilon -= self.epsilon_decay
