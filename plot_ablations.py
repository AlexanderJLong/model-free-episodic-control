import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

files = [("alpha_processed.csv", "TIME-SIG"), ("alpha_processed.csv", "TIME-SIG"),
         ("alpha_processed.csv", "TIME-SIG"), ]
fig, ax = plt.subplots(1, len(files), sharey=True)
sns.set(rc={'figure.figsize': (5, 4)})
sns.set_context("notebook")
sns.set_style("ticks")
for f, ax in zip(files, ax):
    ss = pd.read_csv(f"./results/{f[0]}")
    print(ss)
    compare_var = f[1]
    max_frames = 80_000
    num_games = ss["ENV"].nunique()
    ss[compare_var] = ss[compare_var].astype('category')
    num_lines = ss[compare_var].nunique()

    sns.lineplot("Step",
                 "normalized_reward",
                 ci=0,
                 estimator=np.median,
                 data=ss,
                 linewidth=1.5,
                 hue=compare_var,
                 palette=sns.color_palette("colorblind", num_lines),
                 ax=ax
                 )
    ax.plot((0, max_frames), (0.161, 0.161), c="k", linewidth=2, ls="--", )
    # label="DE-Rainbow")
    ax.plot((0, max_frames), (0.134, 0.134), c="k", linewidth=2, ls=":", )
    # label="SimPLe")
    ax.plot((0, max_frames), (0.212, 0.212), c="k", linewidth=2, ls="-", )
    # label="ARK2")
    ax.set_xlim(0, 8e4)
    ax.set_ylabel("Median Human Normalized Reward")

    # Put a legend below current axis
    # plt.legend(by_label.values(), by_label.keys())
    # plt.subplots_adjust(bottom=0.05)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=5).texts[0].set_text("\u03B1")
    #ax.ylabel("Median Human Normalized Reward")
    #ax.title("Time-Sensitivity")
    #ax.yscale("linear")
    #ax.xlabel("Interactions")

plt.gcf().set_size_inches(12, 5)
plt.tight_layout()
plt.savefig(f"./plots/mhns.png")
plt.savefig(f"./plots/mhns.pdf", format="pdf")
plt.show()
