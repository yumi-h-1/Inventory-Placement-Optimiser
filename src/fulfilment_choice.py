"""Multinomial logistic regression of fulfilment-hub assignment.

Each delivered Olist order item was shipped from one of three seller
regions (pseudo-fulfilment hubs). An MNL recovers this real allocation
policy: P(hub | customer state, product group, basket features), using
alternative-specific hub distances plus case-specific features.

The averaged probabilities form the current-policy flow matrix
P(hub | state) that the optimiser is benchmarked against.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

HUB_LIST = ["HUB_SaoPaulo", "HUB_South", "HUB_MG_RJ"]
SEED = 42


def load():
    orders = pd.read_csv("data/processed/order_items.csv", parse_dates=["week"])
    dist = pd.read_csv("data/processed/state_hub_distance_km.csv",
                       index_col="customer_state")
    return orders, dist


def make_design(orders: pd.DataFrame, dist: pd.DataFrame):
    X = pd.DataFrame(index=orders.index)
    for h in HUB_LIST:                      # alternative-specific distances
        X[f"dist_{h}"] = orders.customer_state.map(dist[h]) / 1000.0  # in 1000 km
    X["log_price"] = np.log1p(orders.price)
    X = pd.concat([X,
                   pd.get_dummies(orders.product_group, prefix="g"),
                   pd.get_dummies(orders.customer_state, prefix="st")], axis=1)
    y = orders.hub.map({h: i for i, h in enumerate(HUB_LIST)}).values
    return X.astype(float), y


def fit(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    model = LogisticRegression(max_iter=3000, C=1.0)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    shares = np.bincount(ytr) / len(ytr)          # share-baseline: predict
    base = np.tile(shares, (len(yte), 1))          # global hub shares
    metrics = {
        "accuracy": accuracy_score(yte, model.predict(Xte)),
        "majority_accuracy": max(np.bincount(yte)) / len(yte),
        "log_loss": log_loss(yte, proba),
        "share_baseline_log_loss": log_loss(yte, base),
    }
    return model, metrics


def flow_matrix(model, orders, dist) -> pd.DataFrame:
    """Observed-feature average of P(hub | order) per customer state."""
    X, _ = make_design(orders, dist)
    proba = model.predict_proba(X)
    out = pd.DataFrame(proba, columns=HUB_LIST)
    out["customer_state"] = orders.customer_state.values
    return out.groupby("customer_state").mean()


def empirical_flows(orders) -> pd.DataFrame:
    return (orders.groupby(["customer_state", "hub"]).size()
            .unstack(fill_value=0).pipe(lambda d: d.div(d.sum(axis=1), axis=0))
            [HUB_LIST])


if __name__ == "__main__":
    orders, dist = load()
    X, y = make_design(orders, dist)
    model, metrics = fit(X, y)
    print({k: round(v, 3) for k, v in metrics.items()})

    flows = flow_matrix(model, orders, dist)
    flows.to_csv("data/processed/routing_probabilities.csv")
    emp = empirical_flows(orders)
    print("\nMNL flow matrix P(hub | state):\n", flows.round(3))
    print("\nmax |MNL - empirical| per state:\n",
          (flows - emp).abs().max(axis=1).round(3))
