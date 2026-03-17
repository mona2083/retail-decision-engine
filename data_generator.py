import numpy as np
import pandas as pd

PRODUCTS = {
    "milk":  {"ja": "牛乳",           "en": "Milk",          "base_price": 3.99, "unit_cost": 2.20, "elasticity": -1.8, "base_demand": 120, "icon": "🥛"},
    "bread": {"ja": "パン",           "en": "Bread",         "base_price": 2.49, "unit_cost": 1.10, "elasticity": -1.4, "base_demand": 90,  "icon": "🍞"},
    "oj":    {"ja": "オレンジジュース", "en": "Orange Juice",  "base_price": 4.99, "unit_cost": 2.80, "elasticity": -2.2, "base_demand": 70,  "icon": "🍊"},
}

WEEKDAY_FACTORS = [0.85, 0.80, 0.90, 1.00, 1.15, 1.40, 1.30]  # Mon-Sun

SEASONAL_FACTORS = {
    "milk":  [1.05, 1.00, 0.95, 0.90, 0.90, 0.92, 0.95, 1.00, 1.05, 1.10, 1.10, 1.08],
    "bread": [1.10, 1.05, 1.00, 0.95, 0.95, 0.95, 1.00, 1.00, 1.05, 1.05, 1.10, 1.15],
    "oj":    [1.20, 1.15, 1.05, 0.90, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20],
}


def generate_weekly_sales(n_weeks: int = 104, random_state: int = 42) -> pd.DataFrame:
    rng   = np.random.RandomState(random_state)
    start = pd.Timestamp("2023-01-02")
    dates = pd.date_range(start, periods=n_weeks, freq="W-MON")
    rows  = []
    for product_id, info in PRODUCTS.items():
        base      = info["base_demand"] * 7
        sf        = SEASONAL_FACTORS[product_id]
        trend     = np.linspace(1.0, 1.08, n_weeks)
        for i, date in enumerate(dates):
            month  = date.month - 1
            season = sf[month]
            noise  = rng.normal(1.0, 0.06)
            sales  = int(base * season * trend[i] * noise)
            rows.append({
                "date":       date,
                "product_id": product_id,
                "sales":      max(sales, 0),
                "price":      info["base_price"] + rng.uniform(-0.10, 0.10),
            })
    return pd.DataFrame(rows)


def generate_daily_sales(n_days: int = 365, random_state: int = 42) -> pd.DataFrame:
    rng   = np.random.RandomState(random_state)
    start = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows  = []
    for product_id, info in PRODUCTS.items():
        base = info["base_demand"]
        sf   = SEASONAL_FACTORS[product_id]
        for date in dates:
            month   = date.month - 1
            weekday = date.dayofweek
            season  = sf[month]
            wday    = WEEKDAY_FACTORS[weekday]
            noise   = rng.normal(1.0, 0.08)
            sales   = int(base * season * wday * noise)
            rows.append({
                "date":       date,
                "product_id": product_id,
                "sales":      max(sales, 0),
                "weekday":    date.day_name(),
            })
    return pd.DataFrame(rows)
