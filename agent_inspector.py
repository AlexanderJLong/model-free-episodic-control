from mfec.agent import MFECAgent
from mfec.utils import Utils
import gym

ENVIRONMENT = "CartPole-v0"
TITLE = "local"
ENVIRONMENT = "CartPole-v0"
AGENT_PATH = ""
RENDER = False
EPOCHS = 300
FRAMES_PER_EPOCH = 400
EXP_SKIP = 1
EPOCHS_TILL_VIS = 100

ACTION_BUFFER_SIZE = 1_000
K = 1
DISCOUNT = 1
EPSILON = 0
SCALE_HEIGHT, SCALE_WIDTH = (1, 4)

FRAMESKIP = 1  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25
agent = MFECAgent(
    ACTION_BUFFER_SIZE,
    K,
    DISCOUNT,
    EPSILON,
    SCALE_HEIGHT,
    SCALE_WIDTH,
    6,
    range(2),
    1,
    1,
)

agent.qec.buffers[0].add([0, -0.5, 0.5, 0], 100, 0, 0)
agent.qec.buffers[0].add([0, 0.5, -0.5, 0], 100, 0, 0)

agent.qec.buffers[1].add([0, 0.5, 0.5, 0], 1, 0, 0)
agent.qec.buffers[1].add([0, -0.5, -0.5, 0], 1, 0, 0)

agent.qec.plot_scatter()
