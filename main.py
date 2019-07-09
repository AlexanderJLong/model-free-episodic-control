#!/usr/bin/env python3

# from pyvirtualdisplay import Display
#
# display = Display(visible=0, size=(80, 60))
# display.start()


import os
import random
from time import gmtime, strftime

import gym

from mfec.agent import MFECAgent
from mfec.utils import Utils

import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("environment")
args = parser.parse_args()
print(args.environment)

ENVIRONMENT = "CartPole-v0"
AGENT_PATH = ""
RENDER = False
RENDER_SPEED = 0.04

EPOCHS = 10000000
FRAMES_PER_EPOCH = 1000

ACTION_BUFFER_SIZE = 100_000
K = 5
DISCOUNT = 1
EPSILON = 0.005

FRAMESKIP = 1  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25

SCALE_HEIGHT = 84
SCALE_WIDTH = 84
STATE_DIMENSION = 64


def main(SEED):
    # Create agent-directory
    execution_time = strftime("%Y-%m-%d-%H%M%S", gmtime())
    agent_dir = os.path.join("agents", f"{ENVIRONMENT}_{execution_time}_SEED_{SEED}")
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make(ENVIRONMENT)

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
        #agent.save(agent_dir)


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
    with Pool(3) as p:
        p.map(main, [1, 2, 3])
