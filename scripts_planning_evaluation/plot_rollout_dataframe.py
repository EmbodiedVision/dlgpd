import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------
# Define the experiments you want to plot
# (subdirectories in experiments/)
ID_LIST = ["pretrained_1", "pretrained_2", "pretrained_3"]

EXPERIMENTS = [str(id) for id in ID_LIST]
filename = "rollouts_runs_" + "_".join(EXPERIMENTS) + ".png"
# -------------------------------------------

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
os.chdir(MODULE_PATH)

rollout_base_dir = "experiments/dlgpd/{experiment_id}"
df_list = []
for experiment_id in EXPERIMENTS:
    rollout_dir = rollout_base_dir.format(experiment_id=experiment_id)
    df_list.append(pd.read_pickle(os.path.join(rollout_dir, "rollout_dataframe.pkl")))
df = pd.concat(df_list, ignore_index=True)

pal = sns.color_palette("pastel").as_hex()
pal_dark = sns.color_palette("dark").as_hex()
METHODS = {
    "dlgpd_mismatching": "DLGPD, mismatching evidence",
    "dlgpd_matching": "DLGPD, matching evidence",
}
METHOD_COLORS = {"dlgpd_mismatching": pal[6], "dlgpd_matching": pal[3]}
METHOD_COLORS_DARK = {"dlgpd_mismatching": pal_dark[6], "dlgpd_matching": pal_dark[3]}


def return_plot(data, ax):
    x, y, hue = "bucket_size", "return", "method_name"
    # Show each observation with a scatterplot
    sns.stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        dodge=False,
        jitter=True,
        alpha=0.45,
        zorder=1,
        palette=METHOD_COLORS,
        ax=ax,
    )
    # Show the conditional means
    sns.pointplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        dodge=False,
        join=True,
        palette=METHOD_COLORS_DARK,
        markers="d",
        scale=0.75,
        ci=None,
        ax=ax,
    )


VARIATION_NAMES = {
    "standard": "Standard pendulum, m=1",
    "invactions": "Inverted actions",
    "lighterpole": "m=0.2",
    "heavierpole": "m=1.5",
}

legend_kwargs = {"loc": "lower right", "ncol": 1, "bbox_to_anchor": (1, -0.05)}

fig, ax_arr = plt.subplots(nrows=1, ncols=4, figsize=(18, 3), sharex=True, sharey=True)
for variation_name, ax in zip(
    ["standard", "invactions", "lighterpole", "heavierpole"], ax_arr
):
    df_filter = df[df["variation_name"] == variation_name]
    if variation_name == "standard":
        # standard variation rollouts without recollected evidence
        # are still 'matching'
        df_filter["method_name"] = df_filter["method_name"].str.replace(
            "dlgpd_mismatching", "dlgpd_matching"
        )

    if len(df_filter) == 0:
        continue

    n_methods = df_filter["method_name"].nunique()
    return_plot(df_filter, ax)
    ax.set_ylabel("Return (sum of rewards)")
    ax.set_xlabel("Rollouts in evidence / re-training set")
    ax.set_ylim([-1600, 0])
    ax.set_title(VARIATION_NAMES[variation_name])
    if variation_name == "heavierpole":
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[n_methods:]
        labels = labels[n_methods:]
        labels = [METHODS[l] for l in labels]
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
        ax.legend(
            handles,
            labels,
            handletextpad=0,
            columnspacing=1,
            frameon=True,
            **legend_kwargs
        )
    else:
        ax.legend_.remove()

plt.savefig(filename)
