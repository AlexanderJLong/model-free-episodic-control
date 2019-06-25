#!/usr/bin/env python3

import os
import random
from time import gmtime, strftime

import gym

from mfec.agent import MFECAgent
from mfec.utils import Utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("environment")
args = parser.parse_args()
print(args.environment)

ENVIRONMENT = "CartPole-v0"  # More games at: https://gym.openai.com/envs/#atari
AGENT_PATH = ""
RENDER = False
RENDER_SPEED = 0.04

EPOCHS = 10000000
FRAMES_PER_EPOCH = 10000
EPOCH_SAVE_FREQ = 300
SEED = 42

ACTION_BUFFER_SIZE = 100000
K = 11
DISCOUNT = 1
EPSILON = 0.005

FRAMESKIP = 4  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25

STATE_DIMENSION = 16


def main():
    random.seed(SEED)

    # Create agent-directory
    execution_time = strftime("%Y-%m-%d-%H%M%S", gmtime())
    agent_dir = os.path.join("agents", ENVIRONMENT + "_" + execution_time)
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
    for epoch in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode(agent, env)
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        if epoch % EPOCHS is 0:
            agent.save(agent_dir)


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
    main()
