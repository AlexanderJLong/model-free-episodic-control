#!/usr/bin/env python3
from pyvirtualdisplay import Display
display = Display(visible=0, size=(640, 640))
display.start()
"""
HOW TO RUN:
This file will run with the command args provided, or will use the
defaults if not provided, over 3 random seeds, with one thread per run

To do a grid search, you can write a run script that calls this fn
with different arugments. k
"""

import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

import random
from time import gmtime, strftime
import shutil
import gym
import itertools
import numpy as np
from glob import glob

from mfec.agent import MFECAgent
from mfec.utils import Utils

import argparse
from collections import deque
from multiprocessing import Pool

# GLOBAl VARS FIXED FOR EACH RUN
TITLE = "Noautonorm"
EPOCHS_TILL_VIS = 2000
EPOCHS = 3000
FRAMES_PER_EPOCH = 1000

config = {
    "ENV": "CartPoleLong",
    "EXP-SKIP": 1,
    "ACTION-BUFFER-SIZE": 50_000,
    "K": 15,
    "DISCOUNT": 1,
    "EPSILON": 0,
    "EPS-DECAY": 0.005,
    "NORM-FREQ": 0,
    "KERNEL-WIDTH": 1,
    "KERNEL-TYPE": "AVG",
    "STATE-DIM": 64,
    "PROJECTION-TYPE": 3,
    "SEED": [1, 2, 3],
}
"""Projection type:
0: Identity
1: Random gau
2: orthogonal random
3: archoplas
4: good manual
5: invert 1
"""


def main(cfg):
    print(cfg)
    # Create agent-directory
    config_string = ""
    for param in cfg:
        config_string += "_" + param + "=" + str(cfg[param])

    if TITLE:
        agent_dir = os.path.join("agents", TITLE + config_string)
    else:
        execution_time = strftime("%Y-%m-%d-%H%M%S", gmtime())
        agent_dir = os.path.join("agents", f"{execution_time}" + config_string)
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    if cfg["ENV"] == "CartPole":
        env = gym.make("CartPole-v0")

    elif cfg["ENV"] == "CartPoleLong":
        env = gym.make("CartPole-v1")

    elif cfg["ENV"] == "CartPolePixels":
        from cartpole_wrapper import pixels_cropped_wrapper
        env = gym.make("CartPole-v1")
        env = pixels_cropped_wrapper(env, diff=True)

    elif cfg["ENV"] == "Breakout":
        from baselines.common.atari_wrappers import make_atari, wrap_deepmind
        env = make_atari('BreakoutNoFrameskip-v4')
        env = wrap_deepmind(env, frame_stack=True, scale=False)
    elif cfg["ENV"] == "Pong":
        env = gym.make("PongNoFrameskip-v0")
        from baselines.common.atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env, frame_stack=False, scale=False, episode_life=True)


    else:
        raise Exception("Invalid env specified")

    print(env.observation_space.shape)

    agent = MFECAgent(
        buffer_size=cfg["ACTION-BUFFER-SIZE"],
        k=cfg["K"],
        discount=cfg["DISCOUNT"],
        epsilon=cfg["EPSILON"],
        observation_dim=np.prod(env.observation_space.shape),
        state_dimension=cfg["STATE-DIM"],
        actions=range(env.action_space.n),
        seed=cfg["SEED"],
        exp_skip=cfg["EXP-SKIP"],
        autonormalization_frequency=cfg["NORM-FREQ"],
        epsilon_decay=cfg["EPS-DECAY"],
        kernel_type=cfg["KERNEL-TYPE"],
        kernel_width=cfg["KERNEL-WIDTH"],
        projection_type=cfg["PROJECTION-TYPE"],
    )
    run_algorithm(agent, env, utils)


def run_algorithm(agent, env, utils):
    frames_left = 0
    for e in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode(agent, env)
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()

        # agent.save(agent_dir)
        # agent.qec.plot3d(both=False, diff=False)
        # agent.qec.plot_scatter()
        if e > EPOCHS_TILL_VIS:
            agent.qec.plot_scatter()
            agent.qec.plot3d(both=False, diff=False)


def run_episode(agent, env):
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()

    done = False
    while not done:
        action, _ = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)

        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += 1
    agent.train()

    # agent.qec.plot_scatter()
    # agent.qec.plot3d(both=F alse, diff=False)

    return episode_frames, episode_reward


if __name__ == "__main__":
    if TITLE:
        # Clear all dirs with the same title before spawning subprocs.
        base_dirs = glob("./agents/" + TITLE + "*")
        print(base_dirs)
        for d in base_dirs:
            shutil.rmtree(d)

    config_vals = list(config.values())
    for i, val in enumerate(config_vals):
        if type(val) is not list:
            config_vals[i] = [config_vals[i]]
    all_configs = []
    all_values = itertools.product(*config_vals)
    for vals in all_values:
        all_configs.append(dict(zip(config.keys(), vals)))

    #main(all_configs[0])
    #exit()

    with Pool(20) as p:
        p.map(main, all_configs)
