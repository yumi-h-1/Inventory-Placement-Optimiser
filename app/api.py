"""FastAPI service exposing the planning models.

Mirrors how models are delivered to production at work: the pipeline
outputs (forecast, flow matrix) are loaded once at startup, and the
optimiser runs on demand with caller-supplied planning assumptions.

Run locally:   uvicorn app.api:app --reload
Run in Docker: see Dockerfile / README.
"""

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from optimise_placement import (  # noqa: E402
    HUB_LIST, calibrate_shipping_cost, hub_capacities, optimise, policy_cost)

app = FastAPI(
    title="Inventory Placement Optimiser",
    description="Forecast-driven inventory placement on the Olist network: "
                "MNL routing model + linear-programming optimiser.",
    version="1.0.0",
)

STATE: dict = {}


@app.on_event("startup")
def load_artifacts():
    proc = ROOT / "data" / "processed"
    try:
        STATE["orders"] = pd.read_csv(proc / "order_items.csv", parse_dates=["week"])
        STATE["dist"] = pd.read_csv(proc / "state_hub_distance_km.csv",
                                    index_col="customer_state")
        STATE["flows"] = pd.read_csv(proc / "routing_probabilities.csv",
                                     index_col="customer_state")
        STATE["plan"] = pd.read_csv(proc / "planning_demand.csv")
    except FileNotFoundError as e:
        raise RuntimeError(
            "Processed data missing. Run the pipeline in src/ first.") from e
    STATE["ship_a"], STATE["ship_b"] = calibrate_shipping_cost(
        STATE["orders"], STATE["dist"])
    STATE["base_capacity"] = hub_capacities(STATE["orders"])


@app.get("/health")
def health():
    return {"status": "ok", "hubs": HUB_LIST,
            "planning_cells": len(STATE["plan"])}


@app.get("/forecast")
def forecast():
    """Planning-week demand per (customer state, product group)."""
    return STATE["plan"].to_dict(orient="records")


@app.get("/flows")
def flows():
    """MNL-estimated current-policy flow matrix P(hub | customer state)."""
    return STATE["flows"].round(4).reset_index().to_dict(orient="records")


class OptimiseRequest(BaseModel):
    capacity_headroom: float = Field(
        1.5, gt=0, le=5,
        description="Hub capacity as a multiple of historical peak weekly throughput")
    holding_cost: float = Field(
        1.0, ge=0, le=50, description="BRL per item per week held at a hub")


@app.post("/optimise")
def run_optimiser(req: OptimiseRequest):
    import optimise_placement as op
    op.HOLDING_COST = req.holding_cost  # planning assumption override

    cap = {h: v / 1.5 * req.capacity_headroom
           for h, v in STATE["base_capacity"].items()}
    a, b = STATE["ship_a"], STATE["ship_b"]

    try:
        opt_cost, placement = optimise(STATE["plan"], STATE["dist"], cap, a, b)
    except AssertionError:
        raise HTTPException(
            status_code=422,
            detail="Infeasible: capacity too low to serve forecast demand.")
    cur_cost = policy_cost(STATE["plan"], STATE["flows"], STATE["dist"], a, b)

    by_hub = placement.groupby("hub").units.sum().round(0)
    return {
        "current_policy_cost_brl": round(cur_cost, 0),
        "optimised_cost_brl": round(opt_cost, 0),
        "saving_pct": round(100 * (cur_cost - opt_cost) / cur_cost, 2),
        "assumptions": req.model_dump(),
        "weekly_units_by_hub": by_hub.to_dict(),
    }
