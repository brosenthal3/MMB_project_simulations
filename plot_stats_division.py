import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics_results import analyze_folder
import sklearn.linear_model as linear_model

division_stats = analyze_folder("results/division_experiment")

def extract_division_rate(filename):
    # get rate from the file name
    base = os.path.basename(filename)
    rate_str = base.replace("division_rate_", "").replace(".npz", "")
    return float(rate_str)

# function to make a dataframe for all iterations of each parameter combination
def division_stats_to_dataframe(division_stats):
    rows = []

    for filename, trials in division_stats.items():
        rate = extract_division_rate(filename)

        for trial_idx, stats in enumerate(trials):
            vols = stats["volumes"]
            deaths = stats["deaths"]

            row = {
                "division_rate": rate,
                "trial": trial_idx,
                "lowest_y": stats["lowest_y"],
                "mean_vol_type1": np.mean(vols[1]) if vols[1] else 0,
                "mean_vol_type2": np.mean(vols[2]) if vols[2] else 0,
                "deaths_type1": deaths[1],
                "deaths_type2": deaths[2],
            }
            rows.append(row)

    return pd.DataFrame(rows)

df_div = division_stats_to_dataframe(division_stats)

# Create a transformed invasion depth column
df_div["invasion_depth"] = 100 - df_div["lowest_y"]

#####################
# Plot invasion depth
#####################
df_div["division_rate"] = pd.to_numeric(df_div["division_rate"])

mean_df = ( df_div .groupby("division_rate", as_index=False)
            .agg(mean_invasion=("invasion_depth", "mean")) )
X = mean_df["division_rate"]
y = mean_df["mean_invasion"]

# Ensure numeric x-axis
df_div["division_rate"] = pd.to_numeric(df_div["division_rate"])

mean_df = (
    df_div
    .groupby("division_rate", as_index=False)
    .agg(mean_invasion=("invasion_depth", "mean"))
)

mean_df["division_rate"] = pd.to_numeric(mean_df["division_rate"])

def perform_regression(X, y):
    X_reshaped = X.values.reshape(-1, 1)
    model = linear_model.LinearRegression()
    model.fit(X_reshaped, y)
    r2 = model.score(X_reshaped, y)
    return model, r2

# Regression
X = df_div["division_rate"]
y = df_div["invasion_depth"]
model, r2 = perform_regression(X, y)
print(r2)

plt.figure(figsize=(7,5))
# Mean + CI per condition
sns.pointplot(
    data=df_div,
    x="division_rate",
    y="invasion_depth",
    errorbar=("ci", 95),
    join=False,
    color="hotpink",
    capsize=0.15,
    markers="o",
    scale=1.2
)

# Raw data
sns.stripplot(
    data=df_div,
    x="division_rate",
    y="invasion_depth",
    color="black",
    alpha=0.45,
    jitter=0.15,
    size=4
)

plt.ylabel("Invasion depth", fontsize=12)
plt.xlabel("Division rate", fontsize=12)

plt.text(
    0.02, 0.95,
    f"$R^2 = {r2:.3f}$",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top"
)

# plot regression line
x_range = np.linspace(0, 0.004, 100)
y_pred = model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_pred, color='red', linestyle='--', label='Regression Line')

sns.despine()
plt.tight_layout()
plt.show()


# --- Mean + CI (pointplot) ---
# sns.pointplot(
#     data=df_div,
#     x="division_rate",
#     y="invasion_depth",
#     errorbar=("ci", 95),
#     join=False,
#     color="hotpink",
#     capsize=0.15,
#     markers="o",
#     scale=1.2
# )
#
# # --- Raw data overlay ---
# sns.stripplot(
#     data=df_div,
#     x="division_rate",
#     y="invasion_depth",
#     color="black",
#     alpha=0.45,
#     jitter=0.15,
#     size=4
# )
#
# # --- Labels and style ---
# plt.ylabel("Invasion depth", fontsize=12)
# plt.xlabel("Division rate", fontsize=12)
# sns.despine()
# plt.tight_layout()
#
# plt.savefig('results/division_experiment/division_stats/invasiondepth_vs_divisionrate.png')
# plt.show()



#####################
# Plot cancer volume
#####################
plt.figure(figsize=(7,5))
sns.pointplot(
    data=df_div,
    x="division_rate",
    y="mean_vol_type1",
    errorbar = ("ci", 95),
    dodge = True,
    join = False,
    color = 'hotpink',
    capsize = 0.1
)
sns.stripplot(
    data=df_div,
    x="division_rate",
    y="mean_vol_type1",
    color="black",
    alpha = 0.4
)
plt.ylabel("Total volume of cancer cells")
plt.xlabel("Division rate")
plt.tight_layout()
sns.despine()
plt.savefig('results/division_experiment/division_stats/cancer_volume_vs_divisionrate.png')
plt.show()

###################################
# Plot number of cancer cell deaths
###################################
plt.figure(figsize=(7,5))
sns.pointplot(
    data=df_div,
    x="division_rate",
    y="deaths_type1",
    errorbar = ("ci", 95),
    dodge = True,
    join = False,
    color = 'hotpink',
    capsize = 0.1
)
sns.stripplot(
    data=df_div,
    x="division_rate",
    y="deaths_type1",
    color="black",
    alpha = 0.4
)
plt.ylabel("Number of cancer cell deaths")
plt.xlabel("Division rate")
plt.tight_layout()
sns.despine()
plt.savefig('results/division_experiment/division_stats/cancer_deaths_vs_divisionrate.png')
plt.show()

