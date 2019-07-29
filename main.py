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

# GLOBAl VARS FIXED FOR EACH RUN
EPOCHS_TILL_VIS = 100
EPOCHS = 300
FRAMES_PER_EPOCH = 400
TITLE= "ConfigExp"
config = {
    "EXP-SKIP": 1,
    "ACTION-BUFFER-SIZE": 1_000_000,
    "K": 50,
    "DISCOUNT": 1,
    "EPSILON": [0.001, 0.01, 0.1, 0.2],
    "EPS-DECAY": 0.000,
    "NORM-FREQ": 20,
    "KERNEL-WIDTH": 1,
    "SEED": [1, 2, 3]
}


# STATE_DIMENSION = 4


def main(cfg):
    # Create agent-directory
    config_string = ""
    for param in cfg:
        config_string += "_" + param + "=" + str(cfg[param])

    print(config_string)
    if TITLE:
        agent_dir = os.path.join("agents", TITLE + config_string)
    else:
        execution_time = strftime("%Y-%m-%d-%H%M%S", gmtime())
        agent_dir = os.path.join("agents", f"{execution_time}" + config_string)
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make("CartPole-v0")

    print(env.observation_space.shape)

    agent = MFECAgent(
        buffer_size=cfg["ACTION-BUFFER-SIZE"],
        k=cfg["K"],
        discount=cfg["DISCOUNT"],
        epsilon=cfg["EPSILON"],
        height=1,
        width=4,
        state_dimension=4,
        actions=range(env.action_space.n),
        seed=cfg["SEED"],
        exp_skip=cfg["EXP-SKIP"],
        autonormalization_frequency=cfg["NORM-FREQ"],
        epsilon_decay=cfg["EPS-DECAY"],
        kernel_width=cfg["KERNEL-WIDTH"],
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
        episode_frames += 1
        step += 1
    agent.train()

    # agent.qec.plot_scatter()
    # agent.qec.plot3d(both=False, diff=False)

    return episode_frames, episode_reward


if __name__ == "__main__":
    if TITLE:
        # Clear all dirs with the same title before spawning subprocs.
        base_dirs = glob("./agents/" + TITLE + "*")
        print(base_dirs)
        for d in base_dirs:
            shutil.rmtree(d)

    # main(4, 1)
    # exit()
    #

    config_vals = list(config.values())
    for i, val in enumerate(config_vals):
        if type(val) is not list:
            config_vals[i] = [config_vals[i]]
    all_configs = []
    all_values = itertools.product(*config_vals)
    for vals in all_values:
        all_configs.append(dict(zip(config.keys(), vals)))
    print("ALL CGFGS", all_configs)

    with Pool(20) as p:
        p.map(main, all_configs)
