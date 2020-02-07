import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

files = [("new_time.csv", "TIME-SIG", "a)", "\u03B1", "colorblind"),
         ("new_fit.csv", "EXPLORE", "b)", "Interpolation", "colorblind"),
         ("new_rep.csv", "PROJECTION", "c)", "Representation", "colorblind"), ]
fig, ax = plt.subplots(1, len(files), sharey=True)
sns.set(rc={'figure.figsize': (5, 4)})
sns.set_context("notebook")
sns.set_style("ticks")
for f, ax in zip(files, ax):
    compare_var = f[1]

    ss = pd.read_csv(f"./results/{f[0]}")
    ss = ss[["Step", "ENV", "normalized_reward", compare_var]]
    print(ss)
    max_frames = 80_001
    num_games = ss["ENV"].nunique()
    ss[compare_var] = ss[compare_var].astype('category')
    ss = ss.groupby(["ENV", "Step", compare_var], as_index=False).mean()
    num_lines = ss[compare_var].nunique()

    p = sns.color_palette(f[4], num_lines)
    if f[1]=="TIME-SIG":
        p[4] = (0,0,0)
    else:
        p.insert(0, (0, 0, 0))
        p = p[:-1]
    g = sns.lineplot("Step",
                 "normalized_reward",
                 ci=0,
                 estimator=np.median,
                 data=ss,
                 linewidth=1.5,
                 hue=compare_var,
                 palette=p,
                 ax=ax,
                 )
    ax.get_legend().get_texts()[0].set_text(f[3])
    #if f[1]=="PROJECTION":
    #    ax.get_legend().get_texts()[1].set_text("downsampling")

    ax.plot((0, max_frames), (0.161, 0.161), c="k", linewidth=2, ls="--", )
    # label="DE-Rainbow")
    ax.plot((0, max_frames), (0.134, 0.134), c="k", linewidth=2, ls=":", )
    # label="SimPLe")
    ax.set_xlim(0, 8e4)
    ax.set_title(f[2], fontsize=11)
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


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='k', lw=2, ls= "--", label='DE Rainbow'),
    Line2D([0], [0], color='k', lw=2, ls= ":", label='SimPLe'),
 ]

plt.figlegend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=[0.5, 0], borderaxespad=0)
plt.gcf().set_size_inches(12, 5.3)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.17)
#plt.tight_layout()
plt.savefig(f"./plots/ablation.png")
plt.savefig(f"./plots/ablation.pdf", format="pdf")
plt.show()
