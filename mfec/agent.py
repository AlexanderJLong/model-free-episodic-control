#!/usr/bin/env python3

from collections import deque

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

        #print(np.max(self.state), np.min(self.state))

        # Exploration
        if self.rs.random_sample() < self.epsilon:
            # don't change current action
            lookup_results = [
                self.klt.estimate(self.state, action, count_weight=self.count_weight)
                for action in self.actions
            ]
            return self.action, self.state, lookup_results

        # Exploitation
        else:
            lookup_results = np.asarray([
                self.klt.estimate(self.state, action, count_weight=self.count_weight)
                for action in self.actions
            ])
            r_estimates = lookup_results[:, 0]
            count_estimates = lookup_results[:, 1]  # count can be < 1. range
            # assert max(count_bonuses) <= 1
            # dist_bonus = buffer_out[:, 2]

            # dist_bonus -= np.min(dist_bonus) - 0.01
            # dist_bonus = dist_bonus / np.max(dist_bonus)
            # print(r_bonus, count_bonus, dist_bonus)

            total_estimates = r_estimates

            probs = np.zeros_like(self.actions)
            probs[np.where(total_estimates == max(total_estimates))] = 1
            probs = probs / sum(probs)
            self.action = self.rs.choice(self.actions, p=probs)

            count_bonus = 1 / np.sqrt(count_estimates[self.action] + 1)
            return self.action, self.state, r_estimates, count_bonus * self.count_weight

    def get_qas(self, state, action):
        return self.klt.estimate(state, action, count_weight=0)[0]

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
                value = r + self.discount * (lr * R + (1 - lr) * last_qs)
                R = r + self.discount * R

            self.klt.update(
                s,
                experience["action"],
                value,
            )
            last_qs = experience["Qs"][experience["action"]]
        self.klt.solidify_values()

        if False:
            """n-step update code"""
            N = self.learning_rate
            reward_hist = deque([], maxlen=N)  # store recent rewards most recent last. Oldest first
            qsa_hist = deque([0], maxlen=N)  # store q_as most recent last. Oldest first. Want one past the last reward

            for _ in range(len(trace)):
                experience = trace.pop()
                s = experience["state"]
                a = experience["action"]
                r = self.clipper(experience["reward"])

                n_step_r = 0
                reward_hist.append(r)
                for i, r_i in enumerate(list(reward_hist)[::-1]):
                    n_step_r += r_i * np.power(self.discount, i)

                n_step_r += qsa_hist[0] * np.power(self.discount, i + 1)
                self.klt.update(
                    s,
                    a,
                    n_step_r,
                )
                qsa_hist.append(experience["Qs"][a])

        # Decay e exponentially
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(f"eps={self.epsilon:.2f}")


def save(self, save_dir):
    with open(f"{save_dir}/agent.pkl", "wb") as f:
        pkl.dump(self, f)
