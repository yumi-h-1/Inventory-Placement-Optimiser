"""Inventory placement optimisation with PuLP.

Decides how many units of each product category to place at each fulfilment
centre for a peak week, minimising outbound shipping cost + inventory
holding cost subject to FC capacity, while meeting all regional demand.

The optimised plan is benchmarked against the *current policy*: demand
flows to FCs in proportion to the MNL routing probabilities (how the
network behaves today), with placement matching those flows.
"""

import numpy as np
import pandas as pd
import pulp

from simulate_demand import FCS, REGIONS, CATEGORIES, distance_matrix

SHIP_COST_PER_UNIT_DIST = 0.40   # GBP per unit per distance unit
HOLDING_COST = {"essentials": 0.15, "outdoor": 0.25, "gifting": 0.35}  # GBP/unit/week


def expected_weekly_demand(demand_df: pd.DataFrame, weeks) -> pd.DataFrame:
    """Mean weekly demand per (region, category) over the given weeks."""
    sub = demand_df[demand_df["week"].isin(weeks)]
    return sub.groupby(["region", "category"])["demand"].mean().reset_index()


def policy_cost(demand_rc: pd.DataFrame, flows: pd.DataFrame) -> float:
    """Cost of the current policy implied by the MNL routing probabilities."""
    dist = distance_matrix()
    total = 0.0
    for _, row in demand_rc.iterrows():
        r, cat, d = row["region"], row["category"], row["demand"]
        for f in FCS:
            units = d * flows.loc[r, f]
            total += units * (SHIP_COST_PER_UNIT_DIST * dist[(r, f)] + HOLDING_COST[cat])
    return total


def optimise_placement(demand_rc: pd.DataFrame, capacity_scale: float = 1.0):
    """Solve the LP: choose flows x[r, c, f] >= 0 minimising cost s.t.
    (1) each (region, category) demand fully served,
    (2) each FC's total placed units within capacity."""
    dist = distance_matrix()
    prob = pulp.LpProblem("inventory_placement", pulp.LpMinimize)

    keys = [(row["region"], row["category"]) for _, row in demand_rc.iterrows()]
    demand_lookup = {(row["region"], row["category"]): row["demand"]
                     for _, row in demand_rc.iterrows()}

    x = pulp.LpVariable.dicts(
        "x", [(r, c, f) for (r, c) in keys for f in FCS], lowBound=0)

    prob += pulp.lpSum(
        x[(r, c, f)] * (SHIP_COST_PER_UNIT_DIST * dist[(r, f)] + HOLDING_COST[c])
        for (r, c) in keys for f in FCS)

    for (r, c) in keys:  # meet demand
        prob += pulp.lpSum(x[(r, c, f)] for f in FCS) == demand_lookup[(r, c)]

    for f, fp in FCS.items():  # capacity
        prob += pulp.lpSum(x[(r, c, f)] for (r, c) in keys) \
            <= fp["capacity"] * capacity_scale

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[status] == "Optimal", pulp.LpStatus[status]

    placement = pd.DataFrame(
        [(r, c, f, x[(r, c, f)].value()) for (r, c) in keys for f in FCS],
        columns=["region", "category", "fc", "units"])
    return pulp.value(prob.objective), placement


def evaluate_scenarios(model_flows: pd.DataFrame, demand_df: pd.DataFrame,
                       n_scenarios: int = 200, seed: int = 7):
    """Monte Carlo: perturb demand and compare policy vs optimised cost."""
    rng = np.random.default_rng(seed)
    peak_weeks = [48, 49, 50, 51, 52]
    base = expected_weekly_demand(demand_df, peak_weeks)
    rows = []
    for s in range(n_scenarios):
        scen = base.copy()
        scen["demand"] = scen["demand"] * rng.lognormal(0, 0.15, len(scen))
        cur = policy_cost(scen, model_flows)
        opt, _ = optimise_placement(scen)
        # Does the current policy breach any FC capacity in this scenario?
        fc_load = {f: 0.0 for f in FCS}
        for _, row in scen.iterrows():
            for f in FCS:
                fc_load[f] += row["demand"] * model_flows.loc[row["region"], f]
        breach = any(fc_load[f] > FCS[f]["capacity"] for f in FCS)
        rows.append({"scenario": s, "policy_cost": cur, "optimised_cost": opt,
                     "saving_pct": 100 * (cur - opt) / cur,
                     "policy_capacity_breach": breach})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    demand = pd.read_csv("data/weekly_demand.csv")
    flows = pd.read_csv("data/routing_probabilities.csv", index_col="region")

    peak = expected_weekly_demand(demand, [48, 49, 50, 51, 52])
    cur_cost = policy_cost(peak, flows)
    opt_cost, placement = optimise_placement(peak)

    print(f"Current-policy cost (peak week): GBP {cur_cost:,.0f}")
    print(f"Optimised placement cost:        GBP {opt_cost:,.0f}")
    print(f"Saving: {100 * (cur_cost - opt_cost) / cur_cost:.1f}%")

    placement.to_csv("data/optimal_placement.csv", index=False)
    scen = evaluate_scenarios(flows, demand)
    scen.to_csv("data/scenario_results.csv", index=False)
    print(scen["saving_pct"].describe().round(2))
