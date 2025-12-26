import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics_results import analyze_folder

adhesion_stats = analyze_folder("results/adhesion_experiment")

def extract_adhesion_rate(filename):
    # get rate from the file name
    base = os.path.basename(filename)
    rate_str = base.replace("adhesion_rate_", "").replace(".npz", "")
    return float(rate_str)

# function to make a dataframe for all iterations of each parameter combination
def adhesion_stats_to_dataframe(adhesion_stats):
    rows = []

    for filename, trials in adhesion_stats.items():
        rate = extract_adhesion_rate(filename)

        for trial_idx, stats in enumerate(trials):
            vols = stats["volumes"]
            deaths = stats["deaths"]

            row = {
                "adhesion_rate": rate,
                "trial": trial_idx,
                "lowest_y": stats["lowest_y"],
                "mean_vol_type1": np.mean(vols[1]) if vols[1] else 0,
                "mean_vol_type2": np.mean(vols[2]) if vols[2] else 0,
                "deaths_type1": deaths[1],
                "deaths_type2": deaths[2],
            }
            rows.append(row)

    return pd.DataFrame(rows)

df_div = adhesion_stats_to_dataframe(adhesion_stats)




# Create a transformed invasion depth column
df_div["invasion_depth"] = 100 - df_div["lowest_y"]

#####################
# Plot invasion depth
#####################

plt.figure(figsize=(7,5))
sns.pointplot(
    data=df_div,
    x="adhesion_rate",
    y="invasion_depth",
    errorbar=("ci", 95),
    join=False,
    color="hotpink",
   # alpha = 0.8,
    capsize=0.15,
    markers="o",
    scale=1.2
)

# Raw data overlay
sns.stripplot(
    data=df_div,
    x="adhesion_rate",
    y="invasion_depth",
    color="black",
    alpha=0.45,
    jitter=0.15,
    size=4
)


plt.ylabel("Invasion depth", fontsize=12)
plt.xlabel("adhesion rate", fontsize=12)
# plt.title("Invasion depth for different adhesion rates", fontsize=14)

# Improve style
sns.despine()
plt.tight_layout()
plt.savefig('results/adhesion_experiment/adhesion_stats/invasiondepth_vs_adhesionrate.png')
plt.show()


#####################
# Plot cancer volume
#####################
plt.figure(figsize=(7,5))
sns.pointplot(
    data=df_div,
    x="adhesion_rate",
    y="mean_vol_type1",
    errorbar = ("ci", 95),
    dodge = True,
    join = False,
    color = 'hotpink',
    capsize = 0.1
)
sns.stripplot(
    data=df_div,
    x="adhesion_rate",
    y="mean_vol_type1",
    color="black",
    alpha = 0.4
)
plt.ylabel("Total volume of cancer cells")
plt.xlabel("adhesion rate")
plt.tight_layout()
sns.despine()
plt.savefig('results/adhesion_experiment/adhesion_stats/cancer_volume_vs_adhesionrate.png')
plt.show()

###################################
# Plot number of cancer cell deaths
###################################
plt.figure(figsize=(7,5))
sns.pointplot(
    data=df_div,
    x="adhesion_rate",
    y="deaths_type1",
    errorbar = ("ci", 95),
    dodge = True,
    join = False,
    color = 'hotpink',
    capsize = 0.1
)
sns.stripplot(
    data=df_div,
    x="adhesion_rate",
    y="deaths_type1",
    color="black",
    alpha = 0.4
)
plt.ylabel("Number of cancer cell deaths")
plt.xlabel("adhesion rate")
plt.tight_layout()
sns.despine()
plt.savefig('results/adhesion_experiment/adhesion_stats/cancer_deaths_vs_adhesionrate.png')
plt.show()

