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
    "alien",
    "amidar",
    "asterix",
    "road_runner",
]

# GLOBAl VARS FIXED FOR EACH RUN
TITLE = "knn"
EPOCHS_TILL_VIS = 2000
EPOCHS = 3000
FRAMES_PER_EPOCH = 5_000

eval_steps = 10_000
total_steps = 100_000
test_eps = 1

#SEED MUST BE LAST IN LIST
config = {
    "ENV": env_list,
    "ACTION-BUFFER-SIZE": 100_000,
    "K": 16,
    "DISCOUNT": 1,
    "EPSILON": 0.0,
    "EPS-DECAY": 0.01,
    "STATE-DIM": [1280, 5000],
    "DISTANCE": "l2",
    "LAST_FRAME_ONLY": False,
    "STICKY-ACTIONS": False,
    "NORMENV": False,
    "STACKED-STATE": 4,
    "WEIGHTING": "none",
    "WARMUP": 0, # min samples in buffer. Can Remove.
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

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)

    # FIX SEEDING
    np.random.seed(cfg["SEED"])
    random.seed(cfg["SEED"])

    # Create env
    if cfg["LAST_FRAME_ONLY"]:
        from rainbow_env import EnvLastFrameOnly
        env = EnvLastFrameOnly(
            seed=cfg["SEED"],
            game=cfg["ENV"],
            normalize=cfg["NORMENV"],
            weighting=cfg["WEIGHTING"],)
    else:
        from rainbow_env import EnvStacked
        env = EnvStacked(
            seed=cfg["SEED"],
            game=cfg["ENV"],
            sticky_actions=cfg["STICKY-ACTIONS"],
            stacked_states=cfg["STACKED-STATE"])

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
        warmup=cfg["WARMUP"],
        distance=cfg["DISTANCE"],
    )

    env.train()  # turn on episodic life
    observation = env.reset()
    trace = []
    for step in tqdm(list(range(total_steps + 1))):

        if step % eval_steps == 0:
            tqdm.write(test_agent(agent, env, test_eps=test_eps, utils=utils, train_step=step))
            # agent.qec.plot3d(both=True, diff=False)

        # Act, and add
        action, state = agent.choose_action(observation)
        observation, reward, done = env.step(action)
        trace.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
            }
        )

        if done:
            agent.train(trace)

            # Reset agent and environment
            observation = env.reset()


def test_agent(agent, env, test_eps, utils, train_step):
    """
    Test the main agent, as well as its two sub-agents over 1 episode
    """
    # No exploration and no episodic life
    agent.training = False
    env.eval()

    for e in range(test_eps):
        s = env.reset()
        done = False
        R = 0
        while not done:
            a, _ = agent.choose_action(s)
            s, r, done = env.step(a)
            R += r

        utils.end_episode(0, R)
    msg = utils.end_epoch(train_step)
    # Revert to training
    agent.training = True
    env.train()

    return msg


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
