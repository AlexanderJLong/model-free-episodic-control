#!/usr/bin/env python3
from pyvirtualdisplay import Display

display = Display(visible=0, size=(640, 640))
display.start()

from pyvirtualdisplay import Display
import tensorflow as tf
import gym
from tqdm import tqdm
import numpy as np

import numpy as np
import gym

from mfec.agent import MFECAgent

# GLOBAl VARS FIXED FOR EACH RUN
TITLE = "Merged-Q-Vals"

cfg = {"ENV": "CartPoleLong", "EXP-SKIP": 1, "ACTION-BUFFER-SIZE": 100_000, "K": 15, "DISCOUNT": 1, "EPSILON": 0,
       "EPS-DECAY": 0.005, "NORM-FREQ": 0, "KERNEL-WIDTH": 1, "KERNEL-TYPE": "AVG", "STATE-DIM": 64,
       "PROJECTION-TYPE": 3, "SEED": [1, 2, 3], }


class Args:
    # Environment details
    obs_size = [4]
    num_actions = 2

    # Agent parameters
    discount = 1
    n_step = 10
    epsilon = 1
    epsilon_final = 0.01
    epsilon_anneal = 20_000

    # Training parameters
    model = "nn"
    preprocessor = 'default'
    history_len = 0
    replay_memory_size = 100_000
    batch_size = 128
    learning_rate = 0.00005
    learn_step = 1

    # Stored variables
    seed = 0
    save_file = "./myrun.npy"


class CombinedAgent:
    def __init__(self, MFEC, DQN):
        """
        Train MFEC and DQN side by side, combine q-values:
        expose: Q-values for both on get_action
        """
        self.mfec_agent = MFEC
        self.dqn_agent = DQN
        self.rs = np.random.RandomState(0)

    def reset(self, obv, train=True):
        self.dqn_agent.Reset(obv, train)
        self.mfec_agent.train()


    def get_action(self, obv):
        """
        Return action, q_vals_mfec, q_values_dqn
        """
        _, dqn_qs = self.dqn_agent.GetAction()
        _, mfec_qs = self.mfec_agent.choose_action(obv)

        values = np.asarray(dqn_qs) + np.asarray(mfec_qs)
        best_actions = np.argwhere(values == np.max(values)).flatten()

        action = self.rs.choice(best_actions)
        self.mfec_agent.action = action  # try with and without this. This keeps MFEC consistent with combined agent
        return action, mfec_qs, dqn_qs

    def train(self, action, reward, state, terminal):
        self.dqn_agent.Update(action, reward, state, terminal)
        self.mfec_agent.receive_reward(reward)

        if terminal:
            self.mfec_agent.train()

def test_agent(agent, env):
    try:
        state = env.reset(train=False)
    except:
        state = env.reset()
    agent.reset(state, train=False)
    R = 0

    terminal = False
    while not terminal:
        action, _, _ = agent.get_action(state)
        state, reward, terminal, info = env.step(action)
        agent.train(action, reward, state, terminal)
        R += reward
    return R


from DQNAgent import DQNAgent

env = gym.make("CartPole-v1")

with tf.Session() as sess:
    args = Args()
    dqn_agent = DQNAgent(sess, args)
    mfec_agent = MFECAgent(buffer_size=cfg["ACTION-BUFFER-SIZE"], k=cfg["K"], discount=cfg["DISCOUNT"],
                           epsilon=cfg["EPSILON"], observation_dim=np.prod(env.observation_space.shape),
                           state_dimension=cfg["STATE-DIM"], actions=range(env.action_space.n), seed=cfg["SEED"],
                           exp_skip=cfg["EXP-SKIP"], autonormalization_frequency=cfg["NORM-FREQ"],
                           epsilon_decay=cfg["EPS-DECAY"], kernel_type=cfg["KERNEL-TYPE"],
                           kernel_width=cfg["KERNEL-WIDTH"], projection_type=cfg["PROJECTION-TYPE"], )

    agent = CombinedAgent(mfec_agent, dqn_agent)

    sess.run(tf.global_variables_initializer())

    # Set up training variables
    training_iters = 100_000
    display_step = 1000
    test_step = 2000
    test_count = 5
    tests_done = 0
    test_results = []

    # Stats for display
    ep_rewards = []
    ep_reward_last = 0
    mfec_values = []
    dqn_values = []
    q_last = 0
    avr_ep_reward = max_ep_reward = avr_q = 0.0

    # Start Agent
    state = env.reset()
    agent.reset(state)
    rewards = []
    terminal = False

    for step in tqdm(list(range(training_iters)), ncols=80):

        # Act, and add
        action, mfec_qs, dqn_qs = agent.get_action(state)
        state, reward, terminal, info = env.step(action)
        agent.train(action, reward, state, terminal)

        # Bookeeping
        rewards.append(reward)
        mfec_values.append(mfec_qs)
        dqn_values.append(dqn_qs)
        #print(mfec_qs, dqn_qs)

        if terminal:
            # Test after every ep.
            ep_rewards.append(np.sum(rewards))
            rewards = []

            R_s = []
            for i in range(test_count):
                R = test_agent(agent, env)
                R_s.append(R)
            print(np.mean(R_s))
            tests_done += 1
            test_results.append({'step': step,
                                 'scores': R_s,
                                 'average': np.mean(R_s),
                                 'max': np.max(R_s),
                                 'mfec_qs': np.mean(mfec_values),
                                 'dqn_qs': np.mean(dqn_values)})

            # Save to file
            summary = {'params': vars(args), 'tests': test_results}
            np.save(args.save_file, summary)

            # Reset agent and environment
            state = env.reset()
            agent.reset(state)

            mfec_values=[]
            dqn_values=[]
