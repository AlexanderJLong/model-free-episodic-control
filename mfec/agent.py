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
        self.training = False

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

    def choose_action(self, observation):
        # Preprocess and project observation to state
        # print(observation)
        self.state = self.transformer.transform(observation.reshape(1, -1))

        # Exploration
        if self.rs.random_sample() < self.epsilon and self.training:
            self.action = self.rs.choice(self.actions)
            return self.action, self.state, [self.klt.estimate(self.state,
                                                               action,
                                                               count_weight=self.count_weight,
                                                               training=self.training)
                                             for action in self.actions]
        # Exploitation
        else:
            q_values = np.asarray([self.klt.estimate(self.state,
                                                     action,
                                                     count_weight=self.count_weight,
                                                     training=self.training)
                                   for action in self.actions])
            # print(q_values)
            # print([len(buff) for buff in self.qec.buffers])
            probs = np.zeros_like(self.actions)
            probs[np.where(q_values == max(q_values))] = 1
            probs = probs / sum(probs)

            # probs = q_values
            # probs[np.where(probs==0)] = 1
            # probs=probs/sum(probs)

            self.action = self.rs.choice(self.actions, p=probs)
            return self.action, self.state, q_values

    def get_max_value(self, state):
        return np.max([self.klt.estimate(state, action, count_weight=0, training=True)
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
                R = r + self.discount * R
                if self.update_type == "MC":
                    value = (1 - self.learning_rate) * experience["Qs"][experience["action"]] + \
                            self.learning_rate * R
                    # value = R
                elif self.update_type == "TD":
                    value = (1 - self.learning_rate) * experience["Qs"][experience["action"]] + \
                            self.learning_rate * (r + np.max(last_Qs))
                    # print(f"r:{r} val:{value}, current:{experience['Qs'][experience['action']]} target:{r + np.max(
                    # last_Qs)}")

            # print(f"step {i},r:{r} R: {R}, 1-step bellman: {r + self.get_max_value(experience['state'])},
            # value: {value} ")

            self.klt.update(
                experience["state"],
                experience["action"],
                value,
            )

            # last_Qs = experience["Qs"]
        self.klt.solidify_values()

        # Decay e exponentially
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(self.epsilon)
        else:
            self.epsilon = 0.05

    def save(self, save_dir):
        with open(f"{save_dir}/agent.pkl", "wb") as f:
            pkl.dump(self, f)
