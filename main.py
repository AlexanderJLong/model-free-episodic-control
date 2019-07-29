#!/usr/bin/env python3

# from pyvirtualdisplay import Display
##
# display = Display(visible=0, size=(80, 60))
# display.start()

"""
HOW TO RUN:
This file will run with the command args provided, or will use thier
defaults if not provided, over 3 random seeds, with one thread per run

To do a grid search, you can write a run script that calls this fn
with different arugments. k
"""

import os
import random
from time import gmtime, strftime
import shutil
import gym
import itertools

from glob import glob

from mfec.agent import MFECAgent
from mfec.utils import Utils

import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("environment")
args = parser.parse_args()
print(args.environment)

TITLE = "K"
ENVIRONMENT = "CartPole-v0"
AGENT_PATH = ""
RENDER = False
EPOCHS = 300
FRAMES_PER_EPOCH = 400
EXP_SKIP = 1
EPOCHS_TILL_VIS = 100

ACTION_BUFFER_SIZE = 1_000_000
K = 50
DISCOUNT = 1
EPSILON = 0
EPS_DECAY = 0.005
AUTONORM_FREQ = 20
KERNEL_WIDTH = 1

FRAMESKIP = 1  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0


# STATE_DIMENSION = 4


def main(STATE_DIMENSION, SEED, NORM_FREQ, KERNEL_WIDTH, K):
    # Create agent-directory
    execution_time = strftime("%Y-%m-%d-%H%M%S", gmtime())
    config = {
        "NORMFREQ": NORM_FREQ,
        "DIM": STATE_DIMENSION,
        "K": K,
        "EPS": EPSILON,
        "KERNELWIDTH": KERNEL_WIDTH,
        "SEED": SEED,
    }
    config_string = ""
    for param in config:
        config_string += "_" + param + "=" + str(config[param])

    print(config_string)
    if TITLE:
        agent_dir = os.path.join(
            "agents",
            TITLE + config_string
        )

    else:
        agent_dir = os.path.join(
            "agents",
            f"{ENVIRONMENT}_{execution_time}" + config_string
        )
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make(ENVIRONMENT)

    print(env.observation_space.shape)
    SCALE_HEIGHT, SCALE_WIDTH = (1, 4)

    try:
        if AGENT_PATH:
            agent = MFECAgent.load(AGENT_PATH)
        else:
            agent = MFECAgent(
                buffer_size=ACTION_BUFFER_SIZE,
                k=K,
                discount=DISCOUNT,
                epsilon=EPSILON,
                height=SCALE_HEIGHT,
                width=SCALE_WIDTH,
                state_dimension=STATE_DIMENSION,
                actions=range(env.action_space.n),
                seed=SEED,
                exp_skip=EXP_SKIP,
                autonormalization_frequency=NORM_FREQ,
                epsilon_decay=EPS_DECAY,
                kernel_width=KERNEL_WIDTH,
            )
        run_algorithm(agent, agent_dir, env, utils)

    finally:
        utils.close()
        env.close()


def run_algorithm(agent, agent_dir, env, utils):
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

        if e > EPOCHS_TILL_VIS:
            agent.qec.plot_scatter()


def run_episode(agent, env):
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()

    done = False
    step = 0
    while not done:
        action = agent.choose_action(observation, step)
        observation, reward, done, _ = env.step(action)

        agent.receive_reward(reward, step)

        episode_reward += reward
        episode_frames += FRAMESKIP
        step += 1
    agent.train()

    # agent.qec.plot_scatter()
    # agent.qec.plot3d(both=False, diff=False)

    return episode_frames, episode_reward


if __name__ == "__main__":
    if TITLE:
        # Clear all dirs with the same title before spawing subprocs.
        base_dirs = glob("./agents/" + TITLE + "*")
        print(base_dirs)
        for d in base_dirs:
            shutil.rmtree(d)
    # main(4, 1)
    # exit()
    #
    ARG1 = [4]
    ARG2 = [1, 2, 3]
    NORM_FREQ = [10]
    KERNEL_WIDTHS = [3]
    K = [1, 5, 10, 20, 50]
    with Pool(20) as p:
        p.starmap(main, itertools.product(
            ARG1,
            ARG2,
            NORM_FREQ,
            KERNEL_WIDTHS,
            K, )
                  )
