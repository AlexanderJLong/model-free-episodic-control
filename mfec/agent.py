#!/usr/bin/env python3

import cloudpickle as pkl
import numpy as np

from mfec.klt import KLT
from scipy.sparse import csr_matrix

class MFECAgent:
    def __init__(
            self,
            buffer_size,
            k_exp,
            k_act,
            discount,
            epsilon,
            observation_dim,
            state_dimension,
            actions,
            seed,
            epsilon_decay,
            clip_rewards,
            M,
            time_sig,
            projection,
            explore,
    ):
        self.rs = np.random.RandomState(seed)
        self.actions = actions
        self.klt = KLT(actions=self.actions,
                       buffer_size=buffer_size,
                       k_exp=k_exp,
                       k_act=k_act,
                       state_dim=state_dimension,
                       M=M,
                       time_sig=time_sig,
                       explore = explore,
                       seed=seed)
        self.projection = self.create_projection(state_dimension, observation_dim, projection)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action = int
        self.train_count = 0

        if clip_rewards:
            self.clipper = lambda x: np.clip(x, -1, 1)
        else:
            self.clipper = lambda x: x

    def create_projection(self, F, D, projection):
        if projection=="sparse":
            s = np.sqrt(D)
            A = np.zeros([F, D], dtype=np.int8)
            for i in range(F):
                for j in range(D):
                    rand = self.rs.random_sample()
                    if rand <= 1 / (2 * s):
                        A[i][j] = 1
                    elif rand < 1 / s:
                        A[i][j] = -1
            return csr_matrix(A)
        elif projection=="gaussian":
            return self.rs.random_sample([F, D])
        else:
            raise ValueError("Invalid projection type. Options are 'gaussian' or 'sparse'")

    def choose_action(self, observation, time):
        # state = self.transformer.transform(observation.reshape(1, -1))[0]
        state = self.projection.dot(observation.flatten())

        query_results = np.asarray([
            self.klt.estimate(state, action, time)
            for action in self.actions])

        r_estimates = query_results

        if self.rs.random_sample() < self.epsilon:
            # Exploration
            probs=r_estimates
            probs = probs / sum(probs)

            action = self.rs.choice(self.actions, p=probs)
        else:
            # Exploitation
            probs = np.zeros_like(self.actions)
            probs[np.where(r_estimates == max(r_estimates))] = 1
            probs = probs / sum(probs)
            action = self.rs.choice(self.actions, p=probs)

        bonus = 0 #self.B/np.sqrt(c_estimates[action])
        return action, state, bonus, 0

    def train(self, trace):
        R = 0.0
        for i in range(len(trace)):
            experience = trace.pop()
            s = experience["state"]
            a = experience["action"]
            t = experience["time"]
            b = experience["bonus"]

            r = self.clipper(experience["reward"]) + b
            R = r + self.discount * R

            self.klt.update(s, a, R, t)

        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay
            print(f"eps={self.epsilon:.2f}")
        else:
            self.epsilon = 0

    def save(self, save_dir):
        with open(f"{save_dir}/agent.pkl", "wb") as f:
            pkl.dump(self, f)
