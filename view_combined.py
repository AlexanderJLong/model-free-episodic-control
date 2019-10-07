import numpy as np
np_load_old = np.load



# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
log = np.load("./myrun.npy")
print(log.item()["tests"])

print("step, average_reward")
steps = []
r = []
dqn = []
mfec = []
dqn_qs = []
mfec_qs = []
for i in log.item()["tests"]:
    print(i['step'], i['main_rewards'])
    steps.append(i['step']/1e6)
    dqn_qs.append(i['dqn_qs'])
    mfec_qs.append(i['mfec_qs'])
    r.append(i['main_rewards'])
    mfec.append(i['mfec_rewards'])
    dqn.append(i['dqn_rewards'])


import matplotlib.pyplot as plt

# Data for plotting

fig, ax = plt.subplots()
ax.plot(steps, r, label="reward")
ax.plot(steps, np.asarray(dqn_qs)*20, label="dqn q-values")
ax.plot(steps, mfec_qs, label="mfec q-values")
ax.plot(steps, dqn, label="dqn reward")
ax.plot(steps, mfec, label="mfec reward")
plt.legend()
ax.set(xlabel='steps (M)', ylabel='average reward',
       title='Hybrid agent on Cartpole-v1')
ax.grid()

fig.savefig("test.png")
plt.show()
