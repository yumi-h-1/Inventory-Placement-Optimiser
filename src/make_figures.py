"""Generate all figures for the README into results/figures/."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate_demand import FCS, REGIONS

FIG = "results/figures"


def plot_demand_seasonality(demand: pd.DataFrame):
    weekly = demand.groupby(["week", "category"])["demand"].sum().unstack()
    ax = weekly.plot(figsize=(9, 4), linewidth=1.6)
    ax.set(title="Simulated weekly demand by category (2 years)",
           xlabel="Week", ylabel="Units")
    ax.legend(title="Category")
    plt.tight_layout()
    plt.savefig(f"{FIG}/demand_seasonality.png", dpi=150)
    plt.close()


def plot_routing_heatmap(flows: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(flows.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(flows.columns)), flows.columns, rotation=20)
    ax.set_yticks(range(len(flows.index)), flows.index)
    for i in range(flows.shape[0]):
        for j in range(flows.shape[1]):
            v = flows.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=9)
    ax.set_title("MNL-estimated P(FC serves order | region) - current policy")
    fig.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{FIG}/routing_probabilities.png", dpi=150)
    plt.close()


def plot_placement(placement: pd.DataFrame):
    by_fc = placement.groupby(["fc", "category"])["units"].sum().unstack()
    ax = by_fc.plot(kind="bar", stacked=True, figsize=(7, 4))
    caps = [FCS[f]["capacity"] for f in by_fc.index]
    ax.scatter(range(len(caps)), caps, marker="_", s=600, color="red",
               label="capacity", zorder=3)
    ax.set(title="Optimised peak-week inventory placement vs FC capacity",
           xlabel="", ylabel="Units")
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG}/optimal_placement.png", dpi=150)
    plt.close()


def plot_scenarios(scen: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(scen["saving_pct"], bins=25, color="#3b6ea5", edgecolor="white")
    axes[0].axvline(scen["saving_pct"].mean(), color="red", linestyle="--",
                    label=f"mean {scen['saving_pct'].mean():.1f}%")
    axes[0].set(title="Cost saving vs current policy\n(200 demand scenarios)",
                xlabel="Saving (%)", ylabel="Scenarios")
    axes[0].legend()

    breach_rate = 100 * scen["policy_capacity_breach"].mean()
    axes[1].bar(["Current policy", "Optimised plan"], [breach_rate, 0.0],
                color=["#c0504d", "#4f8a4f"])
    axes[1].set(title="Scenarios breaching FC capacity (%)",
                ylabel="% of scenarios")
    for i, v in enumerate([breach_rate, 0.0]):
        axes[1].text(i, v + 0.8, f"{v:.0f}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIG}/scenario_analysis.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    demand = pd.read_csv("data/weekly_demand.csv")
    flows = pd.read_csv("data/routing_probabilities.csv", index_col="region")
    placement = pd.read_csv("data/optimal_placement.csv")
    scen = pd.read_csv("data/scenario_results.csv")

    plot_demand_seasonality(demand)
    plot_routing_heatmap(flows)
    plot_placement(placement)
    plot_scenarios(scen)
    print("Figures written to", FIG)
