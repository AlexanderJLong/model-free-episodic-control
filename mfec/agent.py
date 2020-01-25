#!/usr/bin/env python3

import cloudpickle as pkl
import numpy as np
from sklearn import random_projection

from mfec.klt import KLT


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
            count_weight,
            projection_density,
            update_type,
            learning_rate,
            time_sig,
            distance,
    ):
        self.rs = np.random.RandomState(seed)
        self.actions = actions
        self.count_weight = count_weight
        self.update_type = update_type
        self.learning_rate = learning_rate

        self.klt = KLT(actions=self.actions,
                       buffer_size=buffer_size,
                       k=k,
                       state_dim=state_dimension,
                       obv_dim=observation_dim,
                       distance=distance,
                       lr=learning_rate,
                       time_sig=time_sig,
                       seed=seed)

        self.transformer = random_projection.SparseRandomProjection(n_components=state_dimension, dense_output=True,
                                                                    density=projection_density)
        self.transformer.fit(np.zeros([1, observation_dim]))
        # self.transformer.components_.data[np.where(self.transformer.components_.data < 0)] = -1
        # self.transformer.components_.data[np.where(self.transformer.components_.data > 0)] = 1
        # self.transformer.components_ = self.transformer.components_.astype(np.int8)

        # for r in self.transformer.components_:
        #    print(r)

        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action = int
        self.train_count = 0
        self.stats = StatsRecorder(dim=state_dimension)

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

    def choose_action(self, observation):
        # Preprocess and project observation to state
        # print(observation)
        self.state = self.transformer.transform(observation.reshape(1, -1))[0]
        #self.state = self.state//0
#
        #self.state = self.state.astype(np.int)

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            # don't change current action
            self.action = np.random.choice(self.actions)
            return self.action, self.state, 0

        # Exploitation
        else:
            q_values = [
                self.klt.estimate(self.state, action, count_weight=self.count_weight)
                for action in self.actions
            ]
            buffer_out = np.asarray(q_values)
            r_estimate = buffer_out[:, 0]
            count_bonus = 1 / (buffer_out[:, 1] + 0.01)  # TODO CHANGE TO SQRT

            count_bonus -= np.min(count_bonus) + 0.01
            count_bonus = count_bonus / np.max(count_bonus)

            r_bonus = r_estimate
            r_bonus -= np.min(r_bonus) + 0.01
            r_bonus = r_bonus / np.max(r_bonus)

           # print(r_bonus, count_bonus)

            total_estimate = self.count_weight * count_bonus + r_bonus

            probs = np.zeros_like(self.actions)
            probs[np.where(total_estimate == max(total_estimate))] = 1
            probs = probs / sum(probs)

            self.action = self.rs.choice(self.actions, p=probs)
            return self.action, self.state, r_estimate

    def get_max_value(self, state):
        return np.max([self.klt.estimate(state, action, count_weight=0)
                       for action in self.actions
                       ])

    def get_state_value_and_max_q(self, state):
        vals = [self.klt.estimate(state, action, count_weight=0)
                for action in self.actions]
        return np.mean(vals), np.max(vals)

    def train(self, trace):
        # Takes trace object: a list of dicts {"state", "action", "reward"}
        R = 0.0
        # print(f"len trace {trace}")
        lr = self.learning_rate
        states_list = []
        for i in range(len(trace)):
            experience = trace.pop()
            s = experience["state"]
            states_list.append(s)  # strip last dim
            r = self.clipper(experience["reward"])
            if i == 0:
                # last sample
                R = r
                value = R
            else:
                value = r + self.discount * (lr * R + (1 - lr) * np.mean(last_qs))
                R = r + self.discount * R

            self.klt.update(
                s,
                experience["action"],
                value,
                experience["time"],
            )
            last_qs = experience["Qs"]

        self.stats.update(states_list)

        self.train_count+=1
        if self.train_count % 50 == 0:
            self.klt.update_normalization(mean=self.stats.mean, std=self.stats.std)

        # print(self.stats.mean)
        # Decay e exponentially
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(f"eps={self.epsilon:.2f}")


def save(self, save_dir):
    with open(f"{save_dir}/agent.pkl", "wb") as f:
        pkl.dump(self, f)
