"""Prepare the Olist Brazilian e-commerce dataset for network planning.

Source: Brazilian E-Commerce Public Dataset by Olist (Kaggle,
CC BY-NC-SA 4.0). ~100k real orders, 2016-2018, with customer state,
seller state, product category and freight paid per item.

This module builds the three planning inputs:
1. an order-level table with each item's customer state, fulfilment hub
   (seller region) and product group,
2. weekly demand series per (customer state, product group),
3. state centroids (median of real geolocations) and hub distances.

Fulfilment hubs: sellers cluster into three regions that act as
pseudo-fulfilment-centres - Sao Paulo (71% of items), the South
(PR/SC/RS, 13%) and MG/RJ/ES (12%). Orders shipped from other states
(~3%) are excluded.
"""

import numpy as np
import pandas as pd

RAW = "data/olist"
OUT = "data/processed"

HUBS = {
    "HUB_SaoPaulo": ["SP"],
    "HUB_South": ["PR", "SC", "RS"],
    "HUB_MG_RJ": ["MG", "RJ", "ES"],
}
TOP_STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF"]

GROUPS = {
    "home": ["bed_bath_table", "furniture_decor", "housewares", "home_confort",
             "home_construction", "furniture_living_room", "furniture_bedroom",
             "kitchen_dining_laundry_garden_furniture", "garden_tools", "home_appliances",
             "home_appliances_2", "small_appliances", "air_conditioning"],
    "beauty_health": ["health_beauty", "perfumery", "diapers_and_hygiene", "baby"],
    "tech": ["computers_accessories", "telephony", "electronics", "watches_gifts",
             "consoles_games", "audio", "tablets_printing_image", "computers",
             "fixed_telephony", "pc_gamer", "cine_photo"],
}
START, END = "2017-01-02", "2018-08-19"  # trim unreliable head/tail weeks


def state_centroids() -> pd.DataFrame:
    geo = pd.read_csv(f"{RAW}/olist_geolocation_dataset.csv",
                      usecols=["geolocation_state", "geolocation_lat", "geolocation_lng"])
    cent = geo.groupby("geolocation_state").median()
    cent.columns = ["lat", "lng"]
    return cent


def haversine(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lng2 - lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def hub_centroids(cent: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for hub, states in HUBS.items():
        sub = cent.loc[[s for s in states if s in cent.index]]
        rows[hub] = sub.mean()
    return pd.DataFrame(rows).T


def build_orders() -> pd.DataFrame:
    orders = pd.read_csv(f"{RAW}/olist_orders_dataset.csv",
                         usecols=["order_id", "customer_id", "order_status",
                                  "order_purchase_timestamp"],
                         parse_dates=["order_purchase_timestamp"])
    orders = orders[orders.order_status == "delivered"]
    items = pd.read_csv(f"{RAW}/olist_order_items_dataset.csv",
                        usecols=["order_id", "product_id", "seller_id",
                                 "price", "freight_value"])
    cust = pd.read_csv(f"{RAW}/olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_state"])
    sellers = pd.read_csv(f"{RAW}/olist_sellers_dataset.csv",
                          usecols=["seller_id", "seller_state"])
    prods = pd.read_csv(f"{RAW}/olist_products_dataset.csv",
                        usecols=["product_id", "product_category_name"])
    trans = pd.read_csv(f"{RAW}/product_category_name_translation.csv")

    df = (items
          .merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id")
          .merge(cust, on="customer_id")
          .merge(sellers, on="seller_id")
          .merge(prods, on="product_id", how="left")
          .merge(trans, on="product_category_name", how="left"))

    state_to_hub = {s: h for h, ss in HUBS.items() for s in ss}
    df["hub"] = df.seller_state.map(state_to_hub)
    df = df.dropna(subset=["hub"])
    df = df[df.customer_state.isin(TOP_STATES)]

    cat_to_group = {c: g for g, cs in GROUPS.items() for c in cs}
    df["product_group"] = df.product_category_name_english.map(cat_to_group).fillna("other")

    df["week"] = df.order_purchase_timestamp.dt.to_period("W-SUN").dt.start_time
    df = df[(df.week >= START) & (df.week <= END)]

    keep = df[["week", "customer_state", "hub", "product_group",
               "price", "freight_value"]].copy()
    return keep


def weekly_demand(orders: pd.DataFrame) -> pd.DataFrame:
    g = (orders.groupby(["week", "customer_state", "product_group"])
         .size().rename("demand").reset_index())
    # complete the grid so every series has every week (fill 0)
    weeks = pd.date_range(orders.week.min(), orders.week.max(), freq="7D")
    idx = pd.MultiIndex.from_product(
        [weeks, TOP_STATES, list(GROUPS) + ["other"]],
        names=["week", "customer_state", "product_group"])
    return (g.set_index(["week", "customer_state", "product_group"])
            .reindex(idx, fill_value=0).reset_index())


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)

    cent = state_centroids()
    hubs = hub_centroids(cent)
    dist = pd.DataFrame(
        {h: haversine(cent.loc[TOP_STATES, "lat"].values,
                      cent.loc[TOP_STATES, "lng"].values,
                      hubs.loc[h, "lat"], hubs.loc[h, "lng"])
         for h in hubs.index}, index=TOP_STATES)
    dist.rename_axis("customer_state").to_csv(f"{OUT}/state_hub_distance_km.csv")

    orders = build_orders()
    orders.to_csv(f"{OUT}/order_items.csv", index=False)
    demand = weekly_demand(orders)
    demand.to_csv(f"{OUT}/weekly_demand.csv", index=False)

    print(f"order items: {len(orders):,} | weeks: {demand.week.nunique()}")
    print("\nitems per hub:\n", orders.hub.value_counts())
    print("\ndistance (km) customer state -> hub:\n", dist.round(0))
