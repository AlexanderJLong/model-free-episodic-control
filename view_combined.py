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
for i in log.item()["tests"]:
    print(i['step'], i['average'])
    steps.append(i['step']/1e6)
    dqn.append(i['dqn_qs'])
    mfec.append(i['mfec_qs'])
    r.append(i['average'])

import matplotlib.pyplot as plt

# Data for plotting

fig, ax = plt.subplots()
ax.plot(steps, r, label="reward")
ax.plot(steps, dqn, label="dqn q-values")
ax.plot(steps, mfec, label="mfec q-values")
plt.legend()
ax.set(xlabel='steps (M)', ylabel='average reward',
       title='Hybrid agent on Cartpole-v1')
ax.grid()

fig.savefig("test.png")
plt.show()
