#!/usr/bin/env python2

import logging
import os
import time

import numpy as np
import scipy.misc  # TODO other method than scipy?
from atari_py.ale_python_interface import ALEInterface  # TODO ale vs gym

from mfec.agent import MFECAgent
from mfec.qec import QEC
from mfec.utils import Utils

# TODO load parameters as json-config
ROM_FILE_NAME = 'qbert.bin'
SAVE_QEC_TABLE = True
QEC_TABLE_PATH = 'results/qbert_v0/qec_10.pkl'

EPOCHS = 10
FRAMES_PER_EPOCH = 20000
FRAMES_PER_ACTION = 4
DISCOUNT = 1.
K = 11
ACTION_BUFFER_SIZE = 1000000  # 1000000

EPSILON = 1.
EPSILON_MIN = .005
EPSILON_DECAY = 10000

SCALE_WIDTH = 84
SCALE_HEIGHT = 84
STATE_DIMENSION = 64

DISPLAY_SCREEN = False
SEED = 42

ale = None
agent = None
utils = None


def main():
    global utils, ale, agent
    logging.basicConfig(level=logging.INFO)
    np.random.seed(SEED)
    utils = create_utils()
    ale = create_ale()
    agent = create_agent()
    run()


def create_utils():
    execution_time = time.strftime("_%m-%d-%H-%M-%S", time.gmtime())
    result_dir = ROM_FILE_NAME.split('.')[0] + execution_time
    results_dir = os.path.join('results', result_dir)
    os.makedirs(results_dir)
    return Utils(results_dir)


# TODO check more variables
def create_ale():
    env = ALEInterface()
    env.setInt('random_seed', SEED)
    env.setBool('display_screen', DISPLAY_SCREEN)
    env.setFloat('repeat_action_probability', 0.)  # DON'T TURN IT ON!
    env.setBool('color_averaging', True)  # TODO compare to max
    env.loadROM(os.path.join('roms', ROM_FILE_NAME))
    return env


def create_agent():
    actions = range(len(ale.getMinimalActionSet()))

    if QEC_TABLE_PATH:
        qec = utils.load_agent(QEC_TABLE_PATH)

    else:
        # TODO is this _projection correct?
        projection = np.random.randn(STATE_DIMENSION,
                                     SCALE_HEIGHT * SCALE_WIDTH).astype(
            np.float32)
        qec = QEC(actions, ACTION_BUFFER_SIZE, K, projection)

    return MFECAgent(qec, DISCOUNT, actions, EPSILON, EPSILON_MIN,
                     EPSILON_DECAY)


def run():
    frames_played = 0
    for epoch in range(1, EPOCHS + 1):
        frames_left = FRAMES_PER_EPOCH

        while frames_left > 0:
            logging.info(
                'Epoch: {}\tFrames: {}/{}'.format(epoch, frames_played,
                                                  FRAMES_PER_EPOCH * EPOCHS))
            frames_episode = run_episode()
            frames_played += frames_episode
            frames_left -= frames_episode

        utils.save_results(epoch)
        if SAVE_QEC_TABLE:
            utils.save_agent(epoch, agent)


def run_episode():
    episode_reward = 0
    frames = 0

    # TODO terminal if dead?
    while not ale.game_over():
        # TODO observation should be the last 4 frames?
        observation = get_observation()
        action = agent.act(observation)
        reward = sum([ale.act(action) for _ in range(FRAMES_PER_ACTION)])

        agent.receive_reward(reward)
        episode_reward += reward
        frames += FRAMES_PER_ACTION

    agent.train()
    ale.reset_game()
    utils.update_results(episode_reward)
    return frames


# TODO make modular and implement VAE
def get_observation():
    observation = ale.getScreenGrayscale()[:, :, 0]
    return scipy.misc.imresize(observation, size=(SCALE_WIDTH, SCALE_HEIGHT))


if __name__ == "__main__":
    main()
