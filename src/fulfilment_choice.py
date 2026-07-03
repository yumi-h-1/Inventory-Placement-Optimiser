"""Multinomial logistic regression model of fulfilment-centre choice.

In real networks, which FC serves an order is the outcome of routing
software reacting to distance, stock availability and congestion. Here we
simulate two years of order-level routing outcomes from a hidden utility
function, then recover the behaviour with a multinomial logistic regression
(MNL) - the classic discrete-choice model used in supply-chain planning.

The fitted MNL gives P(order from region r is served by FC f | features),
which downstream becomes the descriptive "current policy" cost model that
the optimisation step is benchmarked against.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from simulate_demand import FCS, REGIONS, distance_matrix, RNG_SEED

FC_LIST = list(FCS)

# Hidden "true" routing utility coefficients used only to simulate history.
TRUE_BETA = {"distance": -1.1, "stockout": -2.6, "congestion": -0.9}


def simulate_order_history(n_orders: int = 40_000, seed: int = RNG_SEED) -> pd.DataFrame:
    """Simulate order-level routing outcomes from a softmax over utilities."""
    rng = np.random.default_rng(seed)
    dist = distance_matrix()
    region_names = list(REGIONS)
    region_probs = np.array([REGIONS[r]["pop_weight"] for r in region_names])

    regions = rng.choice(region_names, size=n_orders, p=region_probs)
    rows = []
    for r in regions:
        # Per-order FC state: stockout flags and congestion levels.
        stockout = rng.random(len(FC_LIST)) < 0.12
        congestion = rng.uniform(0, 1, len(FC_LIST))
        utils = np.array([
            TRUE_BETA["distance"] * dist[(r, f)]
            + TRUE_BETA["stockout"] * stockout[i]
            + TRUE_BETA["congestion"] * congestion[i]
            for i, f in enumerate(FC_LIST)
        ])
        p = np.exp(utils - utils.max())
        p /= p.sum()
        chosen = rng.choice(len(FC_LIST), p=p)
        for i, f in enumerate(FC_LIST):
            rows.append({
                "region": r, "fc": f,
                "distance": dist[(r, f)],
                "stockout": int(stockout[i]),
                "congestion": congestion[i],
                "chosen": int(i == chosen),
            })
    return pd.DataFrame(rows)


def to_wide(df_long: pd.DataFrame):
    """Pivot alternative-level rows into one row per order for sklearn."""
    n_fc = len(FC_LIST)
    n_orders = len(df_long) // n_fc
    feats, labels = [], []
    arr = df_long[["distance", "stockout", "congestion", "chosen"]].to_numpy()
    for k in range(n_orders):
        block = arr[k * n_fc:(k + 1) * n_fc]
        feats.append(block[:, :3].ravel())          # (dist, stockout, cong) x 3 FCs
        labels.append(int(np.argmax(block[:, 3])))  # index of chosen FC
    cols = [f"{v}_{f}" for f in FC_LIST for v in ("distance", "stockout", "congestion")]
    return pd.DataFrame(feats, columns=cols), np.array(labels)


def fit_mnl(X: pd.DataFrame, y: np.ndarray):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG_SEED, stratify=y)
    model = LogisticRegression(max_iter=2000)  # multinomial by default
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    metrics = {
        "accuracy": accuracy_score(yte, model.predict(Xte)),
        "log_loss": log_loss(yte, proba),
        "baseline_accuracy": max(np.bincount(yte)) / len(yte),
    }
    return model, metrics


def routing_probabilities(model, seed: int = RNG_SEED) -> pd.DataFrame:
    """Expected P(FC | region) under typical conditions, via Monte Carlo
    over stockout/congestion states - the 'current policy' flow matrix."""
    rng = np.random.default_rng(seed + 1)
    dist = distance_matrix()
    rows = []
    for r in REGIONS:
        probs = np.zeros(len(FC_LIST))
        n_draws = 2000
        for _ in range(n_draws):
            stockout = (rng.random(len(FC_LIST)) < 0.12).astype(int)
            congestion = rng.uniform(0, 1, len(FC_LIST))
            x = pd.DataFrame([np.concatenate([
                [dist[(r, f)], stockout[i], congestion[i]]
                for i, f in enumerate(FC_LIST)
            ])], columns=model.feature_names_in_)
            probs += model.predict_proba(x)[0]
        probs /= n_draws
        rows.append({"region": r, **{f: p for f, p in zip(FC_LIST, probs)}})
    return pd.DataFrame(rows).set_index("region")


if __name__ == "__main__":
    history = simulate_order_history()
    X, y = to_wide(history)
    model, metrics = fit_mnl(X, y)
    print({k: round(v, 3) for k, v in metrics.items()})
    flows = routing_probabilities(model)
    flows.to_csv("data/routing_probabilities.csv")
    print("\nP(FC | region) under current policy:")
    print(flows.round(3))
