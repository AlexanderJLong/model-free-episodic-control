from pyvirtualdisplay import Display
import tensorflow as tf
import gym
from tqdm import tqdm
import numpy as np

display = Display(visible=0, size=(1400, 900))
display.start()


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
    learning_rate = 0.0001
    learn_step = 1

    # Stored variables
    seed = 0
    save_file="./myrun.npy"


def test_agent(agent, env):
    try:
        state = env.reset(train=False)
    except:
        state = env.reset()
    agent.Reset(state, train=False)
    R = 0

    terminal = False
    while not terminal:
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        R += reward
    return R


from DQNAgent import DQNAgent

env = gym.make("CartPole-v1")

with tf.Session() as sess:
    args = Args()
    agent = DQNAgent(sess, args)
    sess.run(tf.global_variables_initializer())
    state = env.reset()
    agent.Reset(state)

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
    qs = []
    q_last = 0
    avr_ep_reward = max_ep_reward = avr_q = 0.0

    # Start Agent
    state = env.reset()
    agent.Reset(state)
    rewards = []
    terminal = False

    for step in tqdm(list(range(training_iters)), ncols=80):

        # Act, and add
        action, q_vals = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)

        # Bookeeping
        rewards.append(reward)
        qs.append(q_vals)

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
            test_results.append({'step': step, 'scores': R_s, 'average': np.mean(R_s), 'max': np.max(R_s)})

            # Save to file
            summary = {'params': vars(args), 'tests': test_results}
            np.save(args.save_file, summary)

            # Reset agent and environment
            state = env.reset()
            agent.Reset(state)
