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
            clip_rewards,
            count_weight,
            distance,
    ):
        self.rs = np.random.RandomState(seed)
        self.actions = actions
        self.count_weight = count_weight
        self.qec = QEC(actions=self.actions,
                       buffer_size=buffer_size,
                       k=k,
                       state_dim=state_dimension,
                       distance=distance,
                       seed=seed)

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

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

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
        if self.rs.random_sample() < self.epsilon and self.training:
            self.action = self.rs.choice(self.actions)
            return self.action, self.state, [self.qec.estimate(self.state, action, use_count_exploration=self.training)
                                             for action in self.actions
                                             ]
        # Exploitation
        else:
            estimates = np.asarray(
                [self.qec.estimate(self.state, action, use_count_exploration=self.training)
                 for action in self.actions
                 ])
        Qs = estimates
        # print(rewards)

        # if self.traininQg:
        #    values = rewards
        # else:
        # counts = estimates[:, 1]
        # print(f"raw dists {dists}")
        # counts = np.sqrt(dists)
        # adj_counts = 1 + self.count_weight * counts / sum(counts)  # Convert to [1,count_weight]
        # print(f"processed dists {dists}")
        # print(rewards)
        # if self.training:
        #    # Explor based on dist
        #    vals = rewards + adj_counts
        #    #print("explor")
        # else:
        #    vals = rewards

        # print([len(b) for b in self.qec.buffers])

        maxes = np.where(Qs == max(Qs))
        # if not np.all(np.equal(maxes, np.where(rewards == max(rewards)))): print("different action")
        probs = np.zeros_like(self.actions)
        probs[maxes] = 1
        probs = probs / sum(probs)
        # print(probs)
        # best_actions = np.argwhere(values == np.max(values)).flatten()
        self.action = self.rs.choice(self.actions, p=probs)

        return self.action, self.state, estimates

    def get_max_value(self, state):
        return np.max([self.qec.estimate(state, action, use_count_exploration=self.training)
                       for action in self.actions
                       ])

    def train(self, trace):
        # Takes trace object: a list of dicts {"state", "action", "reward"}
        R = 0.0
        # print(f"len trace {trace}")
        for i in range(len(trace)):
            experience = trace.pop()

            if not i:
                # last sample
                R = experience["reward"]
                value = R
                # print(f"step {i}, R: {R}, current estimate: {experience['Qs'][experience['action']]}, maxQk+1 {0},
                # new estimate: {R}, value: {value} ")

            else:
                r = self.clipper(experience["reward"])
                R += r
                value = 0.5*experience["Qs"][experience["action"]] + 0.5*(0.5 * R + 0.5 * (r + max(last_Qs)))
                # print(f"step {i},r:{r} R: {R}, 1-step bellman: {r + self.get_max_value(experience['state'])},
                # value: {value} ")

            self.qec.update(
                experience["state"],
                experience["action"],
                value,
            )

            last_Qs = experience["Qs"]
        self.qec.solidify_values()

        # Decay e exponentially
        if self.epsilon > 0:
            self.epsilon /= 1 + self.epsilon_decay
            # print(self.epsilon)

    def save(self, results_dir):
        with open(os.path.join(results_dir, "agent.pkl"), "wb") as file:
            pickle.dump(self, file, 2)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
