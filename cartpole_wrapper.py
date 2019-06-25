import numpy as np
import gym
from gym import spaces


class Pixels(gym.ObservationWrapper):
    def __init__(self, env):
        """
        """
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array')

        screen = np.mean(screen, axis=2) / 255


        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width = screen.shape
        screen = screen[int(screen_height * 0.4):int(screen_height * 0.8):3, ::3]
        print(screen.shape)
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)

        return screen

    def observation(self, observation):
        return self.get_screen()


def pixel_state_wrapper(env, greyscale=True, difference=True, scale=True):
    """
    Configure Cartpole to show pixels as the state
    """
    if greyscale:
        env = Pixels(env)
    return env
