#!/usr/bin/env python3

import os.path
from collections import deque
import numpy as np


class Utils:
    def __init__(self, results_dir, history_len):
        self.results_file = open(os.path.join(results_dir, "results.csv"), "w")
        self.results_file.write(
            "Step,Reward\n"
        )
        self.reward_history = deque([0], maxlen=history_len)
        self.episode_reward = 0

    def log_reward(self, r):
        self.episode_reward += r

    def end_episode(self, ):
        """Should always and only be executed at the end of an episode."""
        self.reward_history.append(self.episode_reward)
        self.episode_reward = 0

    def end_epoch(self, step):
        """Save the results for the given epoch in the results-file"""
        results = [
            step,
            np.mean(self.reward_history),
        ]
        self.results_file.write("{},{}\n".format(*results))
        self.results_file.flush()

        message = (
            "\nStep: {}\tReward: {:.2f}\n"
        )
        print(message.format(*results))

    def close(self):
        self.results_file.close()
