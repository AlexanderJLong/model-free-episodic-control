from mfec.agent import MFECAgent

agent = MFECAgent(
    buffer_size=1_000,
    k=2,
    discount=1,
    epsilon=0,
    height=1,
    width=1,
    state_dimension=1,
    actions=range(2),
    seed=1,
    exp_skip=1,
)

agent.klt.buffers[0].add([0, -0.5, 0.5, 0], 50, 0, 0)
agent.klt.buffers[0].add([0, -0.5, -0.5, 0], 10, 0, 0)
agent.klt.buffers[0].add([0, -0.5, 0, 0], 20, 0, 0)

agent.klt.buffers[1].add([0, -0.5, 0.51, 0], 50, 0, 0)
agent.klt.buffers[1].add([0, 0.5, -0.5, 0], 50, 0, 0)

agent.klt.plot_scatter()
