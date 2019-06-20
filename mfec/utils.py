#!/usr/bin/env python3

import os.path
from datetime import datetime


class Utils:
    def __init__(self, results_dir, frames_per_epoch, max_frames):
        self.results_file = open(os.path.join(results_dir, "results.csv"), "w")
        self.results_file.write(
            "time,frames,episodes,reward_avg\n"
        )
        self.frames_per_epoch = frames_per_epoch
        self.max_frames = max_frames
        self.total_frames = 0
        self.total_episodes = 0
        self.start_time = self._get_time_seconds()

    def _get_time_seconds(self):
        return datetime.now().timestamp()

    def end_episode(self, episode_frames, episode_reward):
        """
        Should be always and only executed at the end of an episode.
        Should print the result and save it in log
        """
        self.total_frames += episode_frames
        self.total_episodes += 1
        results = [
            self._get_time_seconds() - self.start_time,
            self.total_frames,
            self.total_episodes,
            round(episode_reward),
        ]
        self.results_file.write("{:.2f},{},{},{}\n".format(*results))
        self.results_file.flush()

        message = (
            "\nTime: {:.2f}\tFrames: {}\tEpisodes: {}\t"
            "Reward-Avg: {}"
        )
        print(message.format(*results))

    def close(self):
        self.results_file.close()
