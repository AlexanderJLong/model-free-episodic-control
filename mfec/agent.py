#!/usr/bin/env python3

import cloudpickle as pkl
import numpy as np
from sklearn import random_projection

from mfec.klt import KLT


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
            projection_density,
            learning_rate,
            quantize,
            distance,
    ):
        self.rs = np.random.RandomState(seed)
        self.actions = actions
        self.count_weight = count_weight
        self.learning_rate = learning_rate
        self.quantize = quantize
        self.k = k

        self.klt = KLT(actions=self.actions,
                       buffer_size=buffer_size,
                       k=k,
                       state_dim=state_dimension,
                       obv_dim=observation_dim,
                       distance=distance,
                       seed=seed)

        self.transformer = random_projection.SparseRandomProjection(n_components=state_dimension, dense_output=True,
                                                                    density=projection_density)
        self.transformer.fit(np.zeros([1, observation_dim]))

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action = int
        self.state = int

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

    def choose_action(self, observation):
        # Preprocess and project observation to state
        self.state = self.transformer.transform(observation.reshape(1, -1))

        lookup_results = np.asarray([
            self.klt.estimate(self.state, action)
            for action in self.actions
        ])
        reward_estimates = lookup_results[:, 0]

        # Tiebreak same rewards randomly
        probs = np.zeros_like(self.actions)
        probs[np.where(reward_estimates == max(reward_estimates))] = 1

        probs = probs / sum(probs)
        self.action = self.rs.choice(self.actions, p=probs)

        return self.action, self.state, reward_estimates, 0

    def get_qas(self, state, action):
        return self.klt.estimate(state, action)[0]

    def get_state_value_and_max_q(self, state):
        vals = [self.klt.estimate(state, action)
                for action in self.actions]
        return np.mean(vals), np.max(vals)

    def train(self, trace):
        # Takes trace object: a list of dicts {"state", "action", "reward"}
        R = 0.0
        # print(f"len trace {trace}")
        #lr = self.learning_rate

        for i in range(len(trace)):
            experience = trace.pop()
            s = experience["state"]
            r = self.clipper(experience["reward"])
            if i == 0:
                # last sample
                R = r
                value = R
            else:
                R = r + self.discount * R
                value = R

            self.klt.update(
                s,
                experience["action"],
                value,
                experience["time"],
            )

        # Decay e exponentially
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(f"eps={self.epsilon:.2f}")


def save(self, save_dir):
    with open(f"{save_dir}/agent.pkl", "wb") as f:
        pkl.dump(self, f)
