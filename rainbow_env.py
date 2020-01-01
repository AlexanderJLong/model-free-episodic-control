# -*- coding: utf-8 -*-
import random
from collections import deque

import atari_py
import cv2
import numpy as np


class Env:
    def __init__(self, seed, game, buffer_size, sticky_actions):
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setInt('max_num_frames_per_episode', int(108e3))
        self.ale.setFloat('repeat_action_probability', 0.25 if sticky_actions else 0)  # Sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = buffer_size  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=buffer_size)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return state / 255

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(np.zeros([84, 84]))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(30):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return np.asarray(self.state_buffer, dtype=np.float32)  # Lower precision is faster

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros([2, 84, 84])
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = np.asarray(frame_buffer).max(0)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            print("awd")
            lives = self.ale.lives()
            if self.lives > lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return np.asarray(self.state_buffer), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class EnvLastFrameOnly:
    def __init__(self, seed, game, normalize, weighting):
        """
        A few changes here from origional prepro - specifically the dtypes of the arrays are now ints.
        Frame buffer is not recreated at every step, is just updated.
        """
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setInt('max_num_frames_per_episode', int(108e3))
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.training = True  # Consistent with model training mode
        self.normalize = normalize
        self.frame_buffer = np.zeros([2, 84, 84], dtype=np.uint8)
        self.weighting = weighting
        self.median = np.median(cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR).flatten())
        self.log_median = np.log(self.median) if self.median != 0 else 0

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        if self.normalize:
            return np.asarray(state / 255, dtype=np.float32)
        else:
            # return np.asarray(state, dtype=self.np_type)
            if self.weighting == "median":
                return state-self.median
            elif self.weighting == "log":
                return 1+np.log(state+1)
            elif self.weighting == "sqrt":
                return np.sqrt(state)
            elif self.weighting == "shifted":
                logged = 1+np.log(state+1)
                return logged-self.log_median
            elif self.weighting == "none":
                return state
            elif self.weighting == "normalized":
                return state/255

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        self.lives = self.ale.lives()
        return self._get_state()

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                self.frame_buffer[0] = self._get_state()
            elif t == 3:
                self.frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = self.frame_buffer.max(0)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if self.lives > lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        #for r in observation:
        #    print(r)
        return observation, reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class EnvStacked(Env):
    def __init__(self, seed, game, sticky_actions, stacked_states):
        super().__init__(seed, game, buffer_size=stacked_states, sticky_actions=sticky_actions)

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return state - 128

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros([2, 84, 84], dtype=np.int8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = np.asarray(frame_buffer).max(0)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        #if self.training:
        #    #Can probably remove this
        #    lives = self.ale.lives()
        #    if self.lives > lives > 0:  # Lives > 0 for Q*bert
        #        self.life_termination = not done  # Only set flag when not truly done
        #        done = True
        #    self.lives = lives
        # Return state, reward, done
        return np.asarray(self.state_buffer), reward, done
