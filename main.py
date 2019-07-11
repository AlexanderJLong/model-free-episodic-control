#!/usr/bin/env python3

from pyvirtualdisplay import Display
#
display = Display(visible=0, size=(80, 60))
display.start()

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

import gym
import itertools

from mfec.agent import MFECAgent
from mfec.utils import Utils

import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("environment")
args = parser.parse_args()
print(args.environment)

TITLE = "singe_param_pixels_and_diff-ds=4-centered-nostack"
ENVIRONMENT = "CartPole-v0"
AGENT_PATH = ""
RENDER = False
RENDER_SPEED = 0.04

EPOCHS = 300
FRAMES_PER_EPOCH = 400

ACTION_BUFFER_SIZE = 100_000
K = 15
DISCOUNT = 1
EPSILON = 0.005

FRAMESKIP = 1  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25



# STATE_DIMENSION = 4


def main(STATE_DIMENSION, SEED):
    # Create agent-directory
    execution_time = strftime("%Y-%m-%d-%H%M%S", gmtime())
    if TITLE:
        agent_dir = os.path.join(
            "agents",
            TITLE + f"_DIM={STATE_DIMENSION}_K={K}_SEED={SEED}"
        )
    else:
        agent_dir = os.path.join(
            "agents",
            f"{ENVIRONMENT}_{execution_time}_DIM={STATE_DIMENSION}_K={K}_SEED={SEED}"
        )
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make(ENVIRONMENT)
    from cartpole_wrapper import pixel_state_wrapper
    env = pixel_state_wrapper(env)

    print(env.observation_space.shape)
    SCALE_HEIGHT, SCALE_WIDTH = env.observation_space.shape

    try:
        if AGENT_PATH:
            agent = MFECAgent.load(AGENT_PATH)
        else:
            agent = MFECAgent(
                ACTION_BUFFER_SIZE,
                K,
                DISCOUNT,
                EPSILON,
                SCALE_HEIGHT,
                SCALE_WIDTH,
                STATE_DIMENSION,
                range(env.action_space.n),
                SEED,
            )
        run_algorithm(agent, agent_dir, env, utils)

    finally:
        utils.close()
        env.close()


def run_algorithm(agent, agent_dir, env, utils):
    frames_left = 0
    for _ in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode(agent, env)
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()
        # agent.save(agent_dir)


def run_episode(agent, env):
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()

    done = False
    while not done:
        action = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)
        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += FRAMESKIP

    agent.train()
    return episode_frames, episode_reward


if __name__ == "__main__":
    ARG1 = [32]
    ARG2 = [1, 2, 3]
    with Pool(20) as p:
        p.starmap(main, itertools.product(ARG1, ARG2))
