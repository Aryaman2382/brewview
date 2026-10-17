"""
Score the blind LLM-summary labels against the model key.

Usage: python3 score_llm_labels.py llm_label_sheet_labelled.csv llm_label_key.csv

Reports, per model: % faithful, % specific, % both; a paired comparison of the two
models on the same (query, cafe) pairs with an exact McNemar test; and whether the
notebook's automatic 'support' metric separates faithful from unfaithful summaries.
"""
import sys

import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

F, S = "faithful (y/n)", "specific to this cafe (y/n)"

lab = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
key = pd.read_csv(sys.argv[2])
lab["item_id"] = lab["item_id"].astype(int)
df = lab.merge(key, on="item_id")
df["model"] = df["model"].str.replace("Qwen/Qwen2.5-", "").str.replace("-Instruct", "")
df["faithful"] = (df[F] == "y").astype(int)
df["specific"] = (df[S] == "y").astype(int)
df["both"] = df.faithful & df.specific

print("=== Per model ===")
summary = df.groupby("model").agg(n=("item_id", "size"), faithful=("faithful", "mean"),
                                  specific=("specific", "mean"), both=("both", "mean"),
                                  mean_support=("support_metric", "mean"))
print(summary.to_string(float_format=lambda x: f"{x:.0%}" if x <= 1 else f"{x:.0f}"))
summary.to_csv("llm_eval_by_model.csv")

print("\n=== Paired comparison (same query and cafe, exact McNemar test) ===")
wide = df.pivot_table(index=["query", "shop"], columns="model",
                      values=["faithful", "specific"], aggfunc="first").dropna()
small, large = sorted(df["model"].unique())
for col in ["faithful", "specific"]:
    a, b = wide[(col, small)], wide[(col, large)]
    only_large, only_small = int(((b == 1) & (a == 0)).sum()), int(((a == 1) & (b == 0)).sum())
    disc = only_large + only_small
    p = binomtest(only_large, disc, 0.5).pvalue if disc else float("nan")
    print(f"{col:9s} {len(wide)} pairs: only {large} {only_large}, only {small} {only_small}, "
          f"same {len(wide) - disc}; McNemar exact p = {p:.3f}")

print("\n=== Does the automatic support metric track faithfulness? ===")
yes, no = df.loc[df.faithful == 1, "support_metric"], df.loc[df.faithful == 0, "support_metric"]
u, p = mannwhitneyu(yes, no, alternative="two-sided")
auc = u / (len(yes) * len(no))
print(f"mean support: faithful {yes.mean():.3f} (n={len(yes)}) vs unfaithful {no.mean():.3f} (n={len(no)})")
print(f"Mann-Whitney p = {p:.3f}; AUC = {auc:.2f} (0.5 = no better than chance)")
