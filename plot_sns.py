from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sota= { #simple, rainbow, human, random
"alien"               :( 616.9     ,739.9	        ,7127.7	     ,227.8	    ),
"amidar"              :( 74.3      ,188.6	        ,1719.5	     ,5.8	    ),
"assault"             :( 527.2     ,431.2	        ,742	     ,222.4	    ),
"asterix"             :( 1128.3    ,470.8	        ,8503.3	     ,210	    ),
"bank_heist"          :( 34.2      ,51	            ,753.1	     ,14.2	    ),
"battle_zone"         :( 4031.2    ,10124.6         ,37187.5	 ,2360	    ),
"boxing"              :( 7.8       ,0.2	            ,12.1	     ,0.1	    ),
"breakout"            :( 16.4      ,1.9	            ,30.5	     ,1.7	    ),
"chopper_command"     :( 979.4     ,861.8	        ,7387.8	     ,811	    ),
"crazy_climber"       :( 62583.6   ,16185.3         ,35829.4	 ,10780.5   ),
"demon_attack"        :( 208.1     ,508	            ,1971	     ,152.1	    ),
"freeway"             :( 16.7      ,27.9	        ,29.6	     ,0	        ),
"frostbite"           :( 236.9     ,866.8	        ,4334.7	     ,65.2	    ),
"gopher"              :( 596.8     ,349.5	        ,2412.5	     ,257.6     ),
"hero"                :( 2656.6    ,6857	        ,30826.4	 ,1027	    ),
"jamesbond"           :( 100.5     ,301.6	        ,302.8	     ,29	    ),
"kangaroo"            :( 51.2      ,779.3	        ,3035	     ,52	    ),
"krull"               :( 2204.8    ,2851.5          ,2665.5	     ,1598	    ),
"kung_fu_master"      :( 14862.5   ,14346.1         ,22736.3	 ,258.5     ),
"ms_pacman"           :( 1480      ,1204.1          ,6951.6	     ,307.3	    ),
"pong"                :( 12.8      ,-19.3	        ,14.6	     ,-20.7	    ),
"private_eye"         :( 35        ,97.8	        ,69571.3	 ,24.9	    ),
"qbert"               :( 1288.8    ,1152.9          ,13455	     ,163.9	    ),
"road_runner"         :( 5640.6    ,9600	        ,7845	     ,11.5	    ),
"seaquest"            :( 683.3     ,354.1	        ,42054.7	 ,68.4	    ),
"up_n_down"           :( 3350.3    ,2877.4	        ,11693.2	 ,533.4	    ),
}

# simple, otrainbow at 100k, human, random
sota = {
    "alien": (405.2, 739.9, 6875.40, 227.8),
    "amidar": (88, 188.6, 1675.80, 5.8),
    "assault": (369.3, 431.2, 1496.40, 222.4),
    "asterix": (1089.5, 470.8, 8503.3, 210),
    "bank_heist": (8.2, 51.0, 734.40, 14.2),
    "battle_zone": (5184.4, 10124.6, 37800.00, 2360),
    "boxing": (9.1, 0.2, 4.3, 0.1),
    "breakout": (12.7, 1.9, 31.80, 1.7),
    "chopper_command": (1246.9, 861.8, 9881.80, 811),
    "crazy_climber": (398217.8, 16185.3, 35410.50, 10780.5),
    "demon_attack": (169.5, 508.0, 3401.30, 152.1),
    "freeway": (20.3, 27.9, 29.6, 0),
    "frostbite": (254.7, 886.8, 4334.7, 65.2),
    "gopher": (771.0, 349.5, 2321.00, 257.6),
    "hero": (1295.1, 6857.0, 25762.50, 1027),
    "jamesbond": (125.3, 301.6, 406.70, 29),
    "kangaroo": (323.1, 779.3, 3035, 52),
    "krull": (4539.9, 2851.5, 2394.60, 1598),
    "kung_fu_master": (17257.2, 14346.1, 22736, 258.5),
    "ms_pacman": (761.8, 1204.1, 15693.40, 307.3),
    "pong": (5.2, -19.3, 9.3, -20.7),
    "private_eye": (58.3, 97.8, 69571, 24.9),
    "qbert": (559.8, 1152.9, 13455, 163.9),
    "road_runner": (5169.4, 9600.0, 7845, 11.5),
    "seaquest": (370.9, 354.1, 20181.80, 68.4),
    "up_n_down": (2152.6, 2877.4, 9082.00, 533.4),
}

"""
structure is:
time, frames, episodes, reward_avg, reward_max

filenames are: ..._K=1_SEED=1
 """

env_dirs = glob("./agents/*SEED=0*/")
envs = sorted(list(set([d.replace("=", ":").split(":")[1] for d in env_dirs])))
data = []
df = pd.DataFrame()
for i, env in enumerate(envs):
    data = []
    base_dirs = glob(f"./agents/ENV={env}*SEED=0*/")
    print(env)
    try:
        for bd in base_dirs:
            base_dir = bd[:-4]  # get current run 15and strip off the seed
            files = glob(base_dir + "*/results.csv")
            for f in files:
                table = pd.read_csv(f, sep=',', header=0)
                f = f.split("/")[-2]

                for param in f.split(":"):
                    if "=" in param:
                        p, v = param.split("=")
                        if p == "SEED":
                            print(f"seed {v}")
                        table.insert(len(table.columns), p, v)
                df = pd.concat([df, table], ignore_index=True)
    except:
        continue

sns.set_context("paper")
#sns.set(style="whitegrid")
#sns.despine()
#sns.set_palette("colorblind")

df["SEED"] = pd.to_numeric(df["SEED"])
df["STATE-DIM"] = pd.to_numeric(df["STATE-DIM"])
df = df.apply(pd.to_numeric, errors='ignore')
num_envs = df["ENV"].nunique()

compare_var = "EXPLORE"
#compare_var = 'STATE-DIM'
#df = df[(df["STATE-DIM"] == 200)]

df.to_csv("results/df.csv")

cols = min(num_envs, 4)
max_frames = max(df["Step"])

lw = 0.75

if True:
    print(df.to_string())
    g = sns.FacetGrid(df, col="ENV", hue=compare_var, col_wrap=cols, sharey=False, )
    g.set(xlim=(0, 8e4))
    try:
        (g.map(sns.lineplot, "Step", "Reward", ci=90, estimator=np.mean, linewidth=lw)).set_titles("{col_name}")
    except:
        (g.map(plt.plot, "Step", "Reward")).set_titles("{col_name}")

    for ax in g.axes.flat:
        env_name = ax.get_title()
        if env_name in sota:
            #ax.plot((0, max_frames), (sota[env_name][0], sota[env_name][0]), c="k", linewidth=1, ls=":",
            #label="SimPLe Baseline")
            ax.plot((0, max_frames), (sota[env_name][1], sota[env_name][1]), c="k", linewidth=lw, ls="--",
                    label="DE-Rainbow Baseline")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())
    legend = plt.figlegend(handles=by_label.values(), labels=by_label.keys(), loc="upper center", ncol=10, bbox_to_anchor=[0.5, 0.98], borderaxespad=0)
    #legend.get_texts()[0].set_text("Sticky Actions Off")
    #legend.get_texts()[1].set_text("Sticky Actions On")
    #g.add_legend()

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.show()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"./plots/full_run.png")
    plt.savefig(f"./plots/full_run.pdf", format="pdf")


# human normalized median performance
ss = df.groupby(["ENV", "Step", compare_var], as_index=False).agg({"Reward": "mean"})
print(ss[ss["Step"]==80_000])
"""
Create a new column by mapping env name to the sota dict, then convert this column of 
tuples to seperate columns and rename.
"""

ss[["simple", "rainbow", "human", "random"]] = pd.DataFrame(ss["ENV"].map(sota).tolist())
ss["reward_rnd_normed"] = ss["Reward"] - ss["random"]
ss["human_rnd_normed"] = ss["human"] - ss["random"]
ss["normalized_reward"] = ss["reward_rnd_normed"] / ss["human_rnd_normed"]

ss.to_csv("results/processed.csv")

num_games = ss["ENV"].nunique()
ss[compare_var] = ss[compare_var].astype('category')
num_lines = ss[compare_var].nunique()

plt.figure()
sns.set(rc={'figure.figsize':(5,4)})
sns.lineplot("Step",
             "normalized_reward",
             ci=90,
             estimator=np.median,
             data=ss,
             linewidth=1.25,
             hue=compare_var,
             palette=sns.color_palette("colorblind", num_lines),
             )
plt.plot((0, max_frames), (0.161, 0.161), c="k", linewidth=lw, ls="--",
         label="Rainbow (DE)")
plt.plot((0, max_frames), (0.098, 0.098), c="k", linewidth=lw, ls=":",
         label="SimPLe")
plt.xlim(0, 8e4)
plt.gcf().set_size_inches(6, 5)
# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=5)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ylabel("Median Human Normalized Reward")
plt.yscale("linear")

plt.savefig(f"./plots/mhns.png")
plt.savefig(f"./plots/mhns.pdf", format="pdf")
plt.show()
