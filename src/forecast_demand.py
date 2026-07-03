"""Weekly demand forecasting per (customer state, product group).

A pooled ridge regression forecasts weekly demand from trend, annual
Fourier seasonality, recent lags and series identity (state x group
one-hots). Evaluated on the final 8 weeks against a seasonal-naive
baseline (mean of the same series' previous 4 weeks).

Outputs the peak-planning forecast used by the optimiser: demand per
(state, group) for the last 4 in-sample weeks + horizon, plus residuals
for bootstrap scenario generation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

HOLDOUT_WEEKS = 8
LAGS = [1, 2, 3, 4]


def make_features(demand: pd.DataFrame) -> pd.DataFrame:
    df = demand.copy()
    df["series"] = df.customer_state + "|" + df.product_group
    df = df.sort_values(["series", "week"])
    t0 = df.week.min()
    df["t"] = (df.week - t0).dt.days / 7.0
    woy = df.week.dt.isocalendar().week.astype(float)
    for k in (1, 2):
        df[f"sin{k}"] = np.sin(2 * np.pi * k * woy / 52)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * woy / 52)
    for lag in LAGS:
        df[f"lag{lag}"] = df.groupby("series").demand.shift(lag)
    dummies = pd.get_dummies(df.series, prefix="s")
    df = pd.concat([df, dummies], axis=1).dropna(subset=[f"lag{l}" for l in LAGS])
    feature_cols = (["t", "sin1", "cos1", "sin2", "cos2"]
                    + [f"lag{l}" for l in LAGS] + list(dummies.columns))
    return df, feature_cols


def fit_and_evaluate(demand: pd.DataFrame):
    df, cols = make_features(demand)
    cutoff = sorted(df.week.unique())[-HOLDOUT_WEEKS]
    train, test = df[df.week < cutoff], df[df.week >= cutoff]

    model = Ridge(alpha=3.0)
    model.fit(train[cols], train.demand)

    pred = np.clip(model.predict(test[cols]), 0, None)
    naive = test[[f"lag{l}" for l in LAGS]].mean(axis=1)

    metrics = {
        "model_MAE": mean_absolute_error(test.demand, pred),
        "seasonal_naive_MAE": mean_absolute_error(test.demand, naive),
    }
    metrics["improvement_pct"] = 100 * (1 - metrics["model_MAE"] / metrics["seasonal_naive_MAE"])

    test = test[["week", "customer_state", "product_group", "demand"]].copy()
    test["forecast"] = pred
    test["residual"] = test.demand - test.forecast
    return model, metrics, test


def planning_forecast(test: pd.DataFrame) -> pd.DataFrame:
    """Average forecast demand per (state, group) over the holdout weeks,
    used as the planning week's expected demand."""
    return (test.groupby(["customer_state", "product_group"])
            .forecast.mean().rename("demand").reset_index())


if __name__ == "__main__":
    demand = pd.read_csv("data/processed/weekly_demand.csv", parse_dates=["week"])
    model, metrics, test = fit_and_evaluate(demand)
    print({k: round(v, 2) for k, v in metrics.items()})
    test.to_csv("data/processed/forecast_holdout.csv", index=False)
    plan = planning_forecast(test)
    plan.to_csv("data/processed/planning_demand.csv", index=False)
    print(f"\nplanning demand rows: {len(plan)}, total weekly units: {plan.demand.sum():.0f}")
