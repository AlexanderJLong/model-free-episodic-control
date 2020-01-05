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

import numpy as np
from tqdm import tqdm

from mfec.agent import MFECAgent
from mfec.utils import Utils

full_env_list = [
    "alien",
    "amidar",
    "assault",
    "asterix",
    "astroids",
    "atlantis",
    "bank_heist",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "bowling",
    "boxing",
    "carnival",
    "centipede",
    "chopper_command",
    "crazy_climber",
    "demon_attack",
    "double_dunk",
    "elevator_action",
    "enduro",
    "fishing_derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice_hockey",
    "jamesbond",
    "journey_escape",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "montezuma_revenge",
    "ms_pacman",
    "name_this_game",
    "phoenix",
    "pitfall",
    "pitfall2",
    "pooyan",
    "private_eye",
    "riverraid",
    "road_runnerr",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space_invaders",
    "star_gunner",
    "tennis",
    "time_pilot",
    "tutankham",
    "up_n_down",
    "video_pinball",
    "zaxxon",
]

env_list = [
    "alien",
    "amidar",
    "assault",
    "asterix",
    "bank_heist",
    "battle_zone",
    "boxing",
    "breakout",
    "chopper_command",
    "crazy_climber",
    "demon_attack",
    "freeway",
    "frostbite",
    "gopher",
    "hero",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "private_eye",
    "qbert",
    "road_runner",
    "seaquest",
    "up_n_down",
]

small_env_list = [
    "breakout",
    #"freeway",
    "ms_pacman",
    "qbert",
    # "seaquest",
    # "crazy_climber",
]

# GLOBAl VARS FIXED FOR EACH RUN
TITLE = "knn"
EPOCHS_TILL_VIS = 2000
EPOCHS = 3000
FRAMES_PER_EPOCH = 5_000

eval_steps = 2_000
total_steps = 200_000

# SEED MUST BE LAST IN LIST
config = {
    "ENV": "qbert",
    "ACTION-BUFFER-SIZE": total_steps,
    "K": 16,
    "DISCOUNT": 1,
    "EPSILON": 0.6,
    "EPS-DECAY": 0.06,
    "STATE-DIM": 64,
    "DISTANCE": "l2",
    "STICKY-ACTIONS": True,
    "STACKED-STATE": 4,
    "CLIP-REWARD": False,
    "COUNT-WEIGHT": 0,
    "PROJECTION-DENSITY": "auto",
    "UPDATE-TYPE": "MC",
    "LR": 1,
    "AGG-DIST": 1,
    "SEED": list(range(3)),
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
    utils = Utils(agent_dir, history_len=10)

    # FIX SEEDING
    np.random.seed(cfg["SEED"])
    random.seed(cfg["SEED"])

    # Create env
    from dopamine_env import create_atari_environment
    env = create_atari_environment(cfg["ENV"].title())

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
        count_weight=cfg["COUNT-WEIGHT"],
        projection_density=cfg["PROJECTION-DENSITY"],
        distance=cfg["DISTANCE"],
        update_type=cfg["UPDATE-TYPE"],
        agg_dist=cfg["AGG-DIST"],
        learning_rate=cfg["LR"],
    )

    env.train()  # turn on episodic life
    observation = env.reset()
    trace = []
    for step in tqdm(list(range(total_steps + 1))):

        if step % eval_steps == 0:
            utils.end_epoch(step)

        # Act, and add
        action, state, q_vals = agent.choose_action(observation)
        observation, reward, done = env.step(action)
        utils.log_reward(reward)
        trace.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "Qs": q_vals,
            }
        )

        if done:
            utils.end_episode()
            agent.train(trace)

            # Reset agent and environment
            observation = env.reset()

    #print("saving...")
    #agent.save("./saves")

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
