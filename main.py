#!/usr/bin/env python3

# from pyvirtualdisplay import Display
#
# display = Display(visible=0, size=(80, 60))
# display.start()

"""
HOW TO RUN:
This file will run with the command args provided, or will use the
defaults if not provided, over 3 random seeds, with one thread per run

To do a grid search, you can write a run script that calls this fn
with different arugments. k
"""

import itertools
import os
import random
import shutil
from glob import glob
from multiprocessing import Pool
import time

import numpy as np
from tqdm import tqdm

from env_names import *
from mfec.agent import MFECAgent
from mfec.utils import Utils

# GLOBAl VARS FIXED FOR EACH RUN
eval_steps = 2_500
total_steps = 100_000
reward_history_len = 5  # At publication time should be 100.

# SEED MUST BE LAST IN LIST
config = {
    "ENV": env_list,
    "ACTION-BUFFER-SIZE": total_steps,
    "K": 8,
    "DISCOUNT": 0.95,
    "EPSILON": 0,
    "EPS-DECAY": 0.05,
    "STATE-DIM": 200,
    "STICKY-ACTIONS": True,
    "FRAMESTACK": 2,
    "CLIP-REWARD": False,
    "PROJECTION-DENSITY": "auto",
    "M": 20,
    "NORM-FREQ": [10, 1e6],
    "TIME-SIG": 100_000,
    "SEED": list(range(5)),
}
"""Projection type:
0: Identity
1: Random gau
2: orthogonal random
3: archoplas
4: very sparse Achlioptas
5: Scikit sparse random auto

Weighting:
0: log
1: sqrt
2: ^1/4
3: std
"""


def main(cfg):
    print(cfg)
    # Create agent-directory
    config_string = ""
    for param in cfg:
        config_string += param + "=" + str(cfg[param]) + ":"

    agent_dir = os.path.join("agents", config_string)
    os.makedirs(agent_dir)

    # Initialize utils and specify reporting params
    utils = Utils(agent_dir, history_len=reward_history_len)

    # FIX SEEDING
    np.random.seed(cfg["SEED"])
    random.seed(cfg["SEED"])

    # Create env
    from dopamine_env import create_atari_environment
    camelcase_title = cfg["ENV"].title().replace("_", "")
    env = create_atari_environment(
        game_name=camelcase_title,
        sticky_actions=cfg["STICKY-ACTIONS"],
        frame_stack=cfg["FRAMESTACK"],
        seed=cfg["SEED"])

    obv_dim = np.prod(env.reset().shape)
    agent = MFECAgent(
        buffer_size=cfg["ACTION-BUFFER-SIZE"],
        k=cfg["K"],
        discount=cfg["DISCOUNT"],
        epsilon=cfg["EPSILON"],
        observation_dim=obv_dim,
        state_dimension=cfg["STATE-DIM"],
        actions=range(len(env.actions)),
        seed=cfg["SEED"],
        epsilon_decay=cfg["EPS-DECAY"],
        clip_rewards=cfg["CLIP-REWARD"],
        projection_density=cfg["PROJECTION-DENSITY"],
        M=cfg["M"],
        norm_freq=cfg["NORM-FREQ"],
    )

    env.train()  # turn on episodic life
    observation = env.reset()
    trace = []
    for step in tqdm(list(range(total_steps))):

        if step % eval_steps == 0 and step:
            utils.end_epoch(step)

        # Act, and add
        action, state, bonus, estimate = agent.choose_action(observation, time=step)
        observation, reward, done, life_lost = env.step(action)
        # print(action)
        # import matplotlib.pyplot as plt
        # for f in observation:
        #  plt.imshow(f, cmap="Greys")
        #
        #  plt.show()
        #env.render(mode="human")
        #time.sleep(0.01)
        utils.log_reward(reward)
        trace.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "bonus": bonus,
                "time": step,
                "estimate": estimate,
            }
        )

        if life_lost:
            # start a new trace
            agent.train(trace)
            trace = []

        no_recent_reward = len(trace) > 500 and not sum([e["reward"] for e in trace[-500:]])
        if no_recent_reward:
            agent.train(trace)
            trace = []
            done = True

        if done:
            utils.end_episode()
            # Reset agent and environment
            observation = env.reset()

        if step == 95_000:
            agent.save("saves")


        if step > 110_000:
            agent.klt.plot3d()


if __name__ == "__main__":

    # Clear all dirs before spawning subprocs.
    base_dirs = glob("./agents/*")
    for d in base_dirs:
        shutil.rmtree(d)

    # If a list of envs, run one after the other
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
