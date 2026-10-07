"""
Step 2: exploratory data analysis.

summarize()  prints statistics, class balance and correlations.
plot_eda()   saves a six-panel overview figure.
explore()    runs both.
"""
import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, LABELS, TARGET, THRESHOLDS


def _class_counts(df):
    return df["signal_class"].value_counts().reindex(LABELS).fillna(0).astype(int)


def _correlation_frame(df):
    num = df.assign(log_distance=np.log10(df["distance_m"]))
    num = num[["distance_m", "log_distance", "frequency_mhz", "tx_power_dbm",
               "tx_gain_dbi", "rx_gain_dbi", TARGET]]
    return num.corr()


def summarize(df):
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("First rows:")
    print(df.head().to_string(index=False), "\n")

    print("Data types:")
    print(df.dtypes.to_string(), "\n")

    print("Missing values per column:")
    print(df.isna().sum().to_string(), "\n")

    print("Summary statistics (numeric):")
    print(df.describe().round(2).T.to_string(), "\n")

    counts = _class_counts(df)
    print("Class distribution:")
    print(pd.DataFrame({"count": counts, "percent": (100 * counts / len(df)).round(1)}).to_string(), "\n")

    print("RSSI by environment (mean, std, min, max):")
    print(df.groupby("environment")[TARGET].agg(["mean", "std", "min", "max"]).round(2).to_string(), "\n")

    corr = _correlation_frame(df)
    print(f"Correlation with {TARGET}:")
    print(corr[TARGET].drop(TARGET).sort_values().round(3).to_string())
    print("\nNote: log_distance correlates more strongly with RSSI than raw distance,")
    print("      which confirms that RSSI is linear in log-distance.\n")


def plot_eda(df, path=None):
    path = path or FIG_DIR / "eda.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    counts = _class_counts(df)
    corr = _correlation_frame(df)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    colors = {"rural": "tab:green", "suburban": "tab:orange", "urban": "tab:red"}

    ax = axes[0, 0]
    ax.hist(df[TARGET], bins=40, color="steelblue", edgecolor="white")
    for t in THRESHOLDS:
        ax.axvline(t, color="k", linestyle="--", linewidth=1)
    ax.set_title("RSSI distribution (dashed = class thresholds)")
    ax.set_xlabel("RSSI (dBm)")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    for e, g in df.groupby("environment"):
        ax.scatter(g["distance_m"], g[TARGET], s=6, alpha=0.4, color=colors[e], label=e)
    ax.set_title("RSSI vs distance (curved)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("RSSI (dBm)")
    ax.legend(markerscale=3)

    ax = axes[0, 2]
    for e, g in df.groupby("environment"):
        ax.scatter(np.log10(g["distance_m"]), g[TARGET], s=6, alpha=0.4, color=colors[e], label=e)
    ax.set_title("RSSI vs log10(distance) (linear)")
    ax.set_xlabel("log10(distance in m)")
    ax.set_ylabel("RSSI (dBm)")
    ax.legend(markerscale=3)

    ax = axes[1, 0]
    names = sorted(df["environment"].unique())
    ax.boxplot([df.loc[df["environment"] == e, TARGET] for e in names])
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_title("RSSI by environment")
    ax.set_ylabel("RSSI (dBm)")

    ax = axes[1, 1]
    ax.bar(LABELS, counts.to_numpy(), color=["#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("Class counts")
    ax.set_ylabel("Count")

    ax = axes[1, 2]
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation matrix")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"EDA figure saved to {path}\n")
    return path


def explore(df):
    summarize(df)
    plot_eda(df)
