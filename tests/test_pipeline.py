"""Tests for the planning pipeline and API.

Run from the repo root:  pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from prepare_data import haversine  # noqa: E402
from optimise_placement import HUB_LIST, optimise, unit_cost  # noqa: E402


def test_haversine_known_distance():
    # Sao Paulo -> Rio de Janeiro is roughly 360 km
    d = haversine(-23.55, -46.63, -22.91, -43.17)
    assert 330 < d < 390


def test_unit_cost_monotonic_in_distance():
    assert unit_cost(14.3, 0.0105, 500) > unit_cost(14.3, 0.0105, 100)


@pytest.fixture
def toy_problem():
    demand = pd.DataFrame({
        "customer_state": ["SP", "RS"],
        "product_group": ["home", "home"],
        "demand": [100.0, 50.0],
    })
    dist = pd.DataFrame(
        {"HUB_SaoPaulo": [0, 831], "HUB_South": [565, 279], "HUB_MG_RJ": [511, 1310]},
        index=["SP", "RS"])
    dist.index.name = "customer_state"
    cap = {h: 1000.0 for h in HUB_LIST}
    return demand, dist, cap


def test_lp_meets_all_demand(toy_problem):
    demand, dist, cap = toy_problem
    cost, placement = optimise(demand, dist, cap, a=14.3, b=0.0105)
    served = placement.groupby(["customer_state", "product_group"]).units.sum()
    assert np.isclose(served[("SP", "home")], 100)
    assert np.isclose(served[("RS", "home")], 50)


def test_lp_prefers_nearest_hub_when_uncapacitated(toy_problem):
    demand, dist, cap = toy_problem
    _, placement = optimise(demand, dist, cap, a=14.3, b=0.0105)
    top = placement.sort_values("units", ascending=False).groupby(
        "customer_state").first()
    assert top.loc["SP", "hub"] == "HUB_SaoPaulo"
    assert top.loc["RS", "hub"] == "HUB_South"


def test_lp_respects_capacity(toy_problem):
    demand, dist, cap = toy_problem
    cap = dict(cap, HUB_SaoPaulo=30.0)   # force overflow away from SP
    _, placement = optimise(demand, dist, cap, a=14.3, b=0.0105)
    assert placement[placement.hub == "HUB_SaoPaulo"].units.sum() <= 30.0 + 1e-6


def test_api_optimise_endpoint():
    processed = ROOT / "data" / "processed" / "planning_demand.csv"
    if not processed.exists():
        pytest.skip("pipeline outputs not present")
    from fastapi.testclient import TestClient
    from app.api import app
    with TestClient(app) as client:
        r = client.post("/optimise", json={"capacity_headroom": 1.5,
                                           "holding_cost": 1.0})
        assert r.status_code == 200
        body = r.json()
        assert body["optimised_cost_brl"] < body["current_policy_cost_brl"]
        assert 0 < body["saving_pct"] < 100
