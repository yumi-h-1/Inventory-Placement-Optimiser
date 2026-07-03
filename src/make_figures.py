"""Generate all README figures into results/figures/."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "results/figures"
HUB_LIST = ["HUB_SaoPaulo", "HUB_South", "HUB_MG_RJ"]
HUB_LABEL = {"HUB_SaoPaulo": "Sao Paulo hub", "HUB_South": "South hub",
             "HUB_MG_RJ": "MG/RJ hub"}


def plot_demand_and_forecast(demand: pd.DataFrame, holdout: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    weekly = demand.groupby(["week", "product_group"]).demand.sum().unstack()
    weekly.plot(ax=axes[0], linewidth=1.4)
    axes[0].set(title="Olist weekly demand by product group",
                xlabel="", ylabel="Items per week")
    axes[0].legend(title="", fontsize=9)

    agg = holdout.groupby("week")[["demand", "forecast"]].sum()
    axes[1].plot(agg.index, agg.demand, marker="o", label="actual")
    axes[1].plot(agg.index, agg.forecast, marker="s", label="forecast")
    axes[1].set(title="Holdout weeks: total demand, forecast vs actual",
                xlabel="", ylabel="Items per week")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()

    plt.savefig(f"{FIG}/demand_and_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_routing_heatmap(flows: pd.DataFrame):
    flows = flows[HUB_LIST]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(flows.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(HUB_LIST)))
    ax.set_xticklabels([HUB_LABEL[h].replace(" hub", "\nhub") for h in HUB_LIST],
                       fontsize=12)
    ax.set_yticks(range(len(flows.index)))
    ax.set_yticklabels(flows.index, fontsize=12)
    ax.set_ylabel("Customer state", fontsize=11)

    for i in range(flows.shape[0]):
        for j in range(flows.shape[1]):
            v = flows.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=11)

    ax.set_title("Probability each hub ships an order, by customer state\n"
                 "(MNL estimate of the current marketplace policy)",
                 fontsize=13, pad=10)
    cbar = fig.colorbar(im, shrink=0.9, pad=0.02)
    cbar.set_label("Probability")
    plt.savefig(f"{FIG}/routing_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_placement(placement: pd.DataFrame, flows: pd.DataFrame,
                   plan: pd.DataFrame):
    cur_share = (plan.merge(flows.reset_index(), on="customer_state")
                 .pipe(lambda d: pd.Series(
                     {h: (d.demand * d[h]).sum() for h in HUB_LIST})))
    opt_share = placement.groupby("hub").units.sum()[HUB_LIST]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(HUB_LIST))
    ax.bar(xs - 0.18, cur_share.values, width=0.36, label="Current policy")
    ax.bar(xs + 0.18, opt_share.values, width=0.36, label="Optimised plan")
    ax.set_xticks(xs)
    ax.set_xticklabels([HUB_LABEL[h] for h in HUB_LIST], fontsize=11)
    ax.set_ylabel("Items per planning week")
    ax.set_title("Weekly volume by hub: current policy vs optimised placement",
                 fontsize=12)
    ax.legend()
    plt.savefig(f"{FIG}/hub_volumes.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_uncertainty(scen: pd.DataFrame, sens: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(scen.saving_pct, bins=25, color="#3b6ea5", edgecolor="white")
    axes[0].axvline(scen.saving_pct.mean(), color="red", linestyle="--",
                    label=f"mean {scen.saving_pct.mean():.1f}%")
    axes[0].set(title="Cost saving vs current policy\n(200 bootstrap demand scenarios)",
                xlabel="Saving (%)", ylabel="Scenarios")
    axes[0].legend()

    axes[1].plot(100 * sens.regional_capacity_frac, sens.saving_pct,
                 marker="o", color="#4f8a4f")
    axes[1].set(title="Saving vs regional hub capacity",
                xlabel="South + MG/RJ capacity (% of headroom level)",
                ylabel="Saving (%)")
    axes[1].grid(alpha=0.3)

    plt.savefig(f"{FIG}/scenario_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    demand = pd.read_csv("data/processed/weekly_demand.csv", parse_dates=["week"])
    holdout = pd.read_csv("data/processed/forecast_holdout.csv", parse_dates=["week"])
    flows = pd.read_csv("data/processed/routing_probabilities.csv",
                        index_col="customer_state")
    placement = pd.read_csv("data/processed/optimal_placement.csv")
    plan = pd.read_csv("data/processed/planning_demand.csv")
    scen = pd.read_csv("data/processed/scenario_results.csv")
    sens = pd.read_csv("data/processed/capacity_sensitivity.csv")

    plot_demand_and_forecast(demand, holdout)
    plot_routing_heatmap(flows)
    plot_placement(placement, flows, plan)
    plot_uncertainty(scen, sens)
    print("figures written to", FIG)
