import pandas as pd

ab_r = pd.read_csv("results/ab_rep_proc.csv")
logr = pd.read_csv("results/processed.csv")

ab_r = ab_r[(ab_r["PROJECTION"] != "sparse")]

pd.concat([ab_r, logr]).to_csv("results/ab_new.csv")