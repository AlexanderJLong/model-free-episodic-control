#!/usr/bin/env python3
from pyvirtualdisplay import Display

display = Display(visible=0, size=(640, 640))
display.start()

import tensorflow as tf
from tqdm import tqdm

import numpy as np
import gym

from mfec.agent import MFECAgent

from collections import deque

# GLOBAl VARS FIXED FOR EACH RUN
cfg = {"ENV": "CartPoleLong",
       "EXP-SKIP": 1,
       "ACTION-BUFFER-SIZE": 1_000_000,
       "K": 7,
       "DISCOUNT": 1,
       "EPSILON": 0,
       "EPS-DECAY": 0.005,
       "NORM-FREQ": 0,
       "KERNEL-WIDTH": 1,
       "KERNEL-TYPE": "AVG",
       "STATE-DIM": 4,
       "PROJECTION-TYPE": 3,
       "SEED": [1, 2, 3], }


class Args:
    obs_size = [4]
    num_actions = 2

    # Agent parameters
    discount = 0.99
    n_step = 1
    epsilon = 0
    epsilon_final = 0
    epsilon_anneal = 5000

    # Training parameters
    model = "nn"
    preprocessor = 'default'
    history_len = 0
    replay_memory_size = 1_000_000
    batch_size = 30
    learning_rate = 0.001
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
        self.mfec_running_diff = deque(maxlen=1000)
        self.dqn_running_diff = deque(maxlen=1000)
        self.weight = 1  # start as an even mix
        self.step = 0
        self.e = 1

    def reset(self, obv, train=True):
        self.dqn_agent.Reset(obv, train)

    def get_action(self, obv):
        """
        Return action, q_vals_mfec, q_values_dqn
        """

        a, dqn_qs = self.dqn_agent.GetAction()
        _, mfec_qs = self.mfec_agent.choose_action(obv)

        mfec_diff = mfec_qs[0] - mfec_qs[1]
        dqn_diff = dqn_qs[0] - dqn_qs[1]

        self.mfec_running_diff.append(mfec_diff)
        self.dqn_running_diff.append(dqn_diff)

        mfec_diff_normalized = mfec_diff / np.std(self.mfec_running_diff)
        dqn_diff_normalized = dqn_diff / np.std(self.dqn_running_diff)
        if np.isnan(mfec_diff_normalized):
            mfec_diff_normalized = 0

        combined_diff = mfec_diff_normalized * self.weight + dqn_diff_normalized * (1 - self.weight)
        # print(self.e)
        self.e -= 1e-4
        if self.e > np.random.rand():
            action = self.rs.choice([0, 1])
        else:
            if combined_diff == 0:
                action = self.rs.choice([0, 1])
            elif combined_diff > 0:
                action = 0
            else:
                action = 1
        return action, mfec_diff_normalized, dqn_diff_normalized

    def train_dqn(self, a, r, s, d):
        # Must be called after each timestep due to internal state
        self.dqn_agent.Update(a, r, s, d)

    def train_mfec(self, trace):
        # Call at end of episode
        # Takes trace object: a list of dicts {"state", "action", "reward"}
        self.mfec_agent.train(trace)


def test_agent(agent, env):
    """
    Test the main agent, as well as its two sub-agents over 1 episode
    """

    main_R = 0
    dqn_R = 0
    mfec_R = 0

    # Combined
    s = env.reset()
    agent.reset(s, train=False)
    d = False
    while not d:
        a, _, _ = agent.get_action(s)
        s, r, d, _ = env.step(a)
        agent.train_dqn(a, r, s, d)
        main_R += r

    # DQN
    s = env.reset()
    agent.dqn_agent.Reset(s, train=False)  # No updates during testing
    done = False
    while not done:
        a, value = agent.dqn_agent.GetAction()
        s, r, done, _ = env.step(a)
        agent.train_dqn(a, r, s, done)
        dqn_R += r

    # MFEC
    s = env.reset()
    done = False
    while not done:
        a, _ = agent.mfec_agent.choose_action(s)
        s, r, done, _ = env.step(a)
        mfec_R += r

    return main_R, mfec_R, dqn_R


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
    test_results = [[], []]

    # trailing test reward
    dqn_trailing = deque(maxlen=10)
    mfec_trailing = deque(maxlen=10)

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
    trace = []
    done = False

    for step in tqdm(list(range(training_iters)), ncols=80):

        # Act, and add
        action, mfec_qs, dqn_qs = agent.get_action(state)
        old_state = state
        state, reward, done, info = env.step(action)
        agent.train_dqn(action, reward, state, done)
        trace.append(
            {
                "state": old_state,
                "action": action,
                "reward": reward,
            }
        )

        # Bookeeping
        rewards.append(reward)
        mfec_values.append(mfec_qs)
        dqn_values.append(dqn_qs)
        test_results[0].append({'step': step,
                                'mfec_qs': mfec_qs,
                                'dqn_qs': dqn_qs,
                                "combined_diff": mfec_qs * agent.weight + dqn_qs * (1 - agent.weight)})

        if done:
            agent.train_mfec(trace)
            trace = []

            # Test after every ep.
            ep_rewards.append(np.sum(rewards))
            rewards = []

            main_rewards = []
            mfec_rewards = []
            dqn_rewards = []
            for i in range(test_count):
                main_r, mfec_r, dqn_r = test_agent(agent, env)
                main_rewards.append(main_r)
                mfec_rewards.append(mfec_r)
                dqn_rewards.append(dqn_r)

            main_reward = np.mean(main_rewards)
            mfec_reward = np.mean(mfec_rewards)
            dqn_reward = np.mean(dqn_rewards)

            # update trailing reward and set weight
            mfec_trailing.append(mfec_reward)
            dqn_trailing.append(dqn_reward)
            agent.weight = np.mean(mfec_trailing) / (np.mean(mfec_trailing) + np.mean(dqn_trailing))

            print(main_reward, mfec_reward, dqn_reward)
            tests_done += 1
            test_results[1].append({'step': step,
                                    'scores': main_rewards,
                                    'main_rewards': main_reward,
                                    'max': np.max(mfec_rewards),
                                    'mfec_rewards': mfec_reward,
                                    'dqn_rewards': dqn_reward,
                                    "weights": agent.weight})

            # Save to file
            summary = {'params': vars(args), 'tests': test_results}
            np.save(args.save_file, summary)

            # Reset agent and environment
            state = env.reset()
            agent.reset(state)

            mfec_values = []
            dqn_values = []
