import pandas as pd

ab_r = pd.read_csv("results/ab_time_proc.csv")
akr2 = pd.read_csv("results/6_no_exp.csv")
print(max(akr2["normalized_reward"]))
akr2["TIME-SIG"] = "AKR2"

pd.concat([ab_r, akr2]).to_csv("results/new_time.csv")