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
            update_type,
            learning_rate,
            agg_dist,
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
                       agg_dist=agg_dist,
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
        self.state = int

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

    def choose_action(self, observation):
        # Preprocess and project observation to state
        # print(observation)
        self.state = self.transformer.transform(observation.reshape(1, -1))

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            # don't change current action
            q_values = [
                self.klt.estimate(self.state, action, count_weight=self.count_weight)
                for action in self.actions
            ]
            return self.action, self.state, q_values

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

        for i in range(len(trace)):
            experience = trace.pop()
            s = experience["state"]
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

        # Decay e exponentially
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(f"eps={self.epsilon:.2f}")


def save(self, save_dir):
    with open(f"{save_dir}/agent.pkl", "wb") as f:
        pkl.dump(self, f)
