"""Simulate weekly regional demand with seasonality and noise.

Generates a synthetic but realistic demand dataset for a small UK-style
fulfilment network: 6 demand regions x 3 product categories x 104 weeks.
Each category has its own seasonal shape (e.g. summer-peaking, Q4-peaking)
plus a mild trend and Poisson noise, mimicking the demand signals a
network-planning team would forecast against.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42

# Demand regions with (x, y) coordinates on an abstract 0-10 grid and a
# population weight that scales their base demand.
REGIONS = {
    "R1_London":     {"coord": (8.0, 2.0), "pop_weight": 0.30},
    "R2_SouthEast":  {"coord": (7.0, 3.0), "pop_weight": 0.18},
    "R3_Midlands":   {"coord": (5.0, 5.0), "pop_weight": 0.16},
    "R4_NorthWest":  {"coord": (3.5, 7.5), "pop_weight": 0.15},
    "R5_NorthEast":  {"coord": (6.0, 8.5), "pop_weight": 0.11},
    "R6_SouthWest":  {"coord": (2.0, 2.5), "pop_weight": 0.10},
}

# Fulfilment centres (FCs) with locations and per-week unit capacity.
FCS = {
    "FC_A_Midlands":  {"coord": (5.0, 5.5), "capacity": 5200},
    "FC_B_SouthEast": {"coord": (7.2, 2.6), "capacity": 4800},
    "FC_C_North":     {"coord": (4.0, 8.0), "capacity": 3600},
}

# Product categories with base weekly demand (units, network-wide) and a
# seasonal profile: amplitude and peak week within a 52-week year.
CATEGORIES = {
    "essentials": {"base": 4200, "amp": 0.10, "peak_week": 2,  "trend": 0.02},
    "outdoor":    {"base": 2400, "amp": 0.55, "peak_week": 28, "trend": 0.05},
    "gifting":    {"base": 2000, "amp": 0.80, "peak_week": 50, "trend": 0.08},
}

N_WEEKS = 104  # two years


def euclidean(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def distance_matrix():
    """Region -> FC distance lookup used across the project."""
    return {
        (r, f): euclidean(REGIONS[r]["coord"], FCS[f]["coord"])
        for r in REGIONS for f in FCS
    }


def simulate_demand(n_weeks: int = N_WEEKS, seed: int = RNG_SEED) -> pd.DataFrame:
    """Return long-format weekly demand: week, region, category, demand."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, n_weeks + 1):
        year_pos = (week - 1) % 52
        year_frac = (week - 1) / 52
        for cat, p in CATEGORIES.items():
            season = 1 + p["amp"] * np.cos(2 * np.pi * (year_pos - p["peak_week"]) / 52)
            level = p["base"] * season * (1 + p["trend"]) ** year_frac
            for region, rp in REGIONS.items():
                lam = max(level * rp["pop_weight"], 0.1)
                demand = rng.poisson(lam)
                rows.append((week, region, cat, demand))
    return pd.DataFrame(rows, columns=["week", "region", "category", "demand"])


if __name__ == "__main__":
    df = simulate_demand()
    df.to_csv("data/weekly_demand.csv", index=False)
    print(df.groupby("category")["demand"].describe().round(1))
    print(f"\nSaved {len(df)} rows to data/weekly_demand.csv")
