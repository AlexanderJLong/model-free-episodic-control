import numpy as np
import gym
from gym import spaces
from collections import deque

class Pixels(gym.ObservationWrapper):
    def __init__(self, env, downsize, centering):
        """
        Origional: 160x600
        """
        gym.ObservationWrapper.__init__(self, env)
        self.ds = downsize
        self.centering = centering
        if centering:
            self.observation_space = spaces.Box(low=0, high=255, shape=(160//self.ds, 360//self.ds), dtype=np.uint16)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(160//self.ds, 600//self.ds), dtype=np.uint16)

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def observation(self, observation=None):
        screen = self.env.render(mode='rgb_array')
        screen = np.asarray(np.mean(screen, axis=2), dtype=np.int16)

        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width = screen.shape
        screen = screen[int(screen_height * 0.4):int(screen_height * 0.8):, ::]

        if self.centering:
            # Convert to float, rescale, convert to torch tensor
            # (this doesn't require a copy)

            view_width = int(screen_width * 0.6)
            cart_location = self.get_cart_location(screen_width)
            if cart_location < view_width // 2:
                slice_range = slice(view_width)
            elif cart_location > (screen_width - view_width // 2):
                slice_range = slice(-view_width, None)
            else:
                slice_range = slice(cart_location - view_width // 2,
                                    cart_location + view_width // 2)
            # Strip off the edges, so that we have a square image centered on a cart
            screen = screen[:, slice_range]

        #downsize
        screen = screen[::self.ds, ::self.ds]
        return screen

class OrigionalPlusDiff(gym.Wrapper):
    def __init__(self, env):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.frames = deque([], maxlen=2)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0]*2, shp[1]), dtype=np.float16)

    def reset(self):
        ob = self.env.reset()
        for _ in range(2):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == 2
        diff = self.frames[-1]-self.frames[-2]
        out = np.concatenate([diff, self.frames[-1]])
        return out


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def pixel_state_wrapper(env, greyscale=True, difference=True, scale=True):
    """
    Configure Cartpole to show pixels as the state
    """
    if greyscale:
        env = Pixels(env, downsize=8, centering=False)
        env = OrigionalPlusDiff(env)
    return env
