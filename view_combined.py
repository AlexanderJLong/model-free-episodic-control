import numpy as np
np_load_old = np.load



# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
log = np.load("./myrun.npy")
print(log.item()["tests"])

print("step, average_reward")
steps = []
q_steps = []
r = []
dqn = []
mfec = []
dqn_qs = []
mfec_qs = []
results = log.item()["tests"]
for i in results[1]:
    print(i['step'], i['main_rewards'])
    steps.append(i['step']/1e6)

    r.append(i['main_rewards'])
    mfec.append(i['mfec_rewards'])
    dqn.append(i['dqn_rewards'])

for i in results[0]:
    q_steps.append(i["step"]/1e6)
    dqn_qs.append(i['dqn_qs'])
    mfec_qs.append(i['mfec_qs'])


import matplotlib.pyplot as plt

# Data for plotting
mfec_qs = np.array(mfec_qs)
dqn_qs = np.array(dqn_qs)

fig, (r_ax, q_ax) = plt.subplots(nrows=2, ncols=1, sharex=True)
r_ax.plot(steps, r, label="reward")
r_ax.plot(steps, dqn, label="dqn reward")
r_ax.plot(steps, mfec, label="mfec reward")
r_ax.legend()

q_ax.plot(q_steps, dqn_qs[:, 0], label="dqn Qa0", linestyle=":")
q_ax.plot(q_steps, dqn_qs[:, 1], label="dqn Qa1", linestyle=":")

q_ax.plot(q_steps, mfec_qs[:, 0], label="mfec Qa0", linestyle=":")
q_ax.plot(q_steps, mfec_qs[:, 1], label="mfec Qa1", linestyle=":")
q_ax.legend()

for ax in [r_ax, q_ax]:
    ax.set(xlabel='steps (M)', ylabel='average reward',
           title='Hybrid agent on Cartpole-v1')
    ax.grid()

#plt.show()
#plt.scatter(dqn_qs, mfec_qs,)
fig.savefig("test.png")
plt.show()
