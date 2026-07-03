"""Inventory placement optimisation on the Olist network with PuLP.

Decides how forecast weekly demand per (customer state, product group)
should be served from the three fulfilment hubs, minimising shipping +
holding cost subject to hub capacity. Shipping cost per item is
calibrated from Olist's real freight charges (freight ~ a + b * km).

Benchmark: the current marketplace policy, i.e. the MNL flow matrix
P(hub | state), which ships most items from Sao Paulo regardless of
customer location.

Uncertainty: 200 scenarios bootstrap the forecaster's holdout residuals
onto the planning demand; each scenario is re-optimised and compared
with current-policy cost and capacity feasibility.
"""

import numpy as np
import pandas as pd
import pulp

HUB_LIST = ["HUB_SaoPaulo", "HUB_South", "HUB_MG_RJ"]
HOLDING_COST = 1.0          # BRL per item per week held at a hub
CAPACITY_HEADROOM = 1.5     # hubs can hold up to 1.5x historical peak weekly throughput
SEED = 7


def calibrate_shipping_cost(orders: pd.DataFrame, dist: pd.DataFrame):
    """freight_value ~ a + b * distance(customer_state, actual hub)."""
    km = np.array([dist.loc[s, h] for s, h in
                   zip(orders.customer_state, orders.hub)])
    b, a = np.polyfit(km, orders.freight_value, 1)
    return float(a), float(b)


def hub_capacities(orders: pd.DataFrame) -> dict:
    weekly = orders.groupby(["week", "hub"]).size().unstack(fill_value=0)
    return (weekly.max() * CAPACITY_HEADROOM).to_dict()


def unit_cost(a, b, km):
    return a + b * km


def policy_cost(demand: pd.DataFrame, flows: pd.DataFrame,
                dist: pd.DataFrame, a: float, b: float) -> float:
    total = 0.0
    for _, row in demand.iterrows():
        for h in HUB_LIST:
            units = row.demand * flows.loc[row.customer_state, h]
            total += units * (unit_cost(a, b, dist.loc[row.customer_state, h])
                              + HOLDING_COST)
    return total


def policy_breaches(demand: pd.DataFrame, flows: pd.DataFrame, cap: dict) -> bool:
    load = {h: 0.0 for h in HUB_LIST}
    for _, row in demand.iterrows():
        for h in HUB_LIST:
            load[h] += row.demand * flows.loc[row.customer_state, h]
    return any(load[h] > cap[h] for h in HUB_LIST)


def optimise(demand: pd.DataFrame, dist: pd.DataFrame,
             cap: dict, a: float, b: float):
    prob = pulp.LpProblem("placement", pulp.LpMinimize)
    keys = list(demand[["customer_state", "product_group"]].itertuples(index=False))
    dem = {(r.customer_state, r.product_group): r.demand
           for r in demand.itertuples()}
    x = pulp.LpVariable.dicts(
        "x", [(s, g, h) for (s, g) in keys for h in HUB_LIST], lowBound=0)

    prob += pulp.lpSum(
        x[(s, g, h)] * (unit_cost(a, b, dist.loc[s, h]) + HOLDING_COST)
        for (s, g) in keys for h in HUB_LIST)
    for (s, g) in keys:
        prob += pulp.lpSum(x[(s, g, h)] for h in HUB_LIST) == dem[(s, g)]
    for h in HUB_LIST:
        prob += pulp.lpSum(x[(s, g, h)] for (s, g) in keys) <= cap[h]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[status] == "Optimal"
    placement = pd.DataFrame(
        [(s, g, h, x[(s, g, h)].value()) for (s, g) in keys for h in HUB_LIST],
        columns=["customer_state", "product_group", "hub", "units"])
    return pulp.value(prob.objective), placement


def scenarios(plan: pd.DataFrame, holdout: pd.DataFrame, flows, dist, cap, a, b,
              n: int = 200, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    res = holdout.groupby(["customer_state", "product_group"]).residual
    res_map = {k: v.values for k, v in res}
    rows = []
    for i in range(n):
        scen = plan.copy()
        scen["demand"] = [
            max(d + rng.choice(res_map[(s, g)]), 0.0)
            for s, g, d in zip(scen.customer_state, scen.product_group, scen.demand)]
        cur = policy_cost(scen, flows, dist, a, b)
        opt, _ = optimise(scen, dist, cap, a, b)
        rows.append({"scenario": i,
                     "policy_cost": cur, "optimised_cost": opt,
                     "saving_pct": 100 * (cur - opt) / cur,
                     "policy_capacity_breach": policy_breaches(scen, flows, cap)})
    return pd.DataFrame(rows)


def capacity_sensitivity(plan, flows, dist, cap, a, b) -> pd.DataFrame:
    """How much of the saving depends on regional hub capacity?
    Scale South and MG_RJ capacity from 0% to 100% of their headroom
    level (Sao Paulo kept unconstrained-high) and re-optimise."""
    rows = []
    cur = policy_cost(plan, flows, dist, a, b)
    for f in np.linspace(0, 1, 11):
        scaled = {"HUB_SaoPaulo": 10 * cap["HUB_SaoPaulo"],
                  "HUB_South": f * cap["HUB_South"],
                  "HUB_MG_RJ": f * cap["HUB_MG_RJ"]}
        opt, _ = optimise(plan, dist, scaled, a, b)
        rows.append({"regional_capacity_frac": f,
                     "saving_pct": 100 * (cur - opt) / cur})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    orders = pd.read_csv("data/processed/order_items.csv", parse_dates=["week"])
    dist = pd.read_csv("data/processed/state_hub_distance_km.csv",
                       index_col="customer_state")
    flows = pd.read_csv("data/processed/routing_probabilities.csv",
                        index_col="customer_state")
    plan = pd.read_csv("data/processed/planning_demand.csv")
    holdout = pd.read_csv("data/processed/forecast_holdout.csv")

    a, b = calibrate_shipping_cost(orders, dist)
    print(f"shipping cost fit: {a:.2f} + {b:.4f} * km  (BRL per item)")
    cap = hub_capacities(orders)
    print("hub capacities (items/week):", {k: int(v) for k, v in cap.items()})

    cur = policy_cost(plan, flows, dist, a, b)
    opt, placement = optimise(plan, dist, cap, a, b)
    print(f"current-policy cost: BRL {cur:,.0f} | optimised: BRL {opt:,.0f} "
          f"| saving {100*(cur-opt)/cur:.1f}%")
    placement.to_csv("data/processed/optimal_placement.csv", index=False)

    sens = capacity_sensitivity(plan, flows, dist, cap, a, b)
    sens.to_csv("data/processed/capacity_sensitivity.csv", index=False)
    print("\nsaving vs regional capacity:\n", sens.round(2).to_string(index=False))

    scen = scenarios(plan, holdout, flows, dist, cap, a, b)
    scen.to_csv("data/processed/scenario_results.csv", index=False)
    print(scen.saving_pct.describe().round(2))
    print("policy capacity breach rate:", scen.policy_capacity_breach.mean())
