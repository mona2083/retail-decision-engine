import numpy as np
from scipy.optimize import minimize_scalar


def demand_at_price(price: float, base_price: float, base_demand: float, elasticity: float) -> float:
    return max(base_demand * (price / base_price) ** elasticity, 0)


def profit_at_price(price: float, base_price: float, base_demand: float, elasticity: float, unit_cost: float) -> float:
    d = demand_at_price(price, base_price, base_demand, elasticity)
    return (price - unit_cost) * d


def find_optimal_price(base_price: float, base_demand: float, elasticity: float, unit_cost: float,
                       min_price: float | None = None, max_price: float | None = None) -> dict:
    if min_price is None:
        min_price = unit_cost * 1.01
    if max_price is None:
        max_price = base_price * 2.5

    def neg_profit(p):
        return -profit_at_price(p, base_price, base_demand, elasticity, unit_cost)

    result        = minimize_scalar(neg_profit, bounds=(min_price, max_price), method="bounded")
    optimal_price = result.x
    optimal_demand = demand_at_price(optimal_price, base_price, base_demand, elasticity)
    optimal_profit = profit_at_price(optimal_price, base_price, base_demand, elasticity, unit_cost)

    return {
        "optimal_price":  round(optimal_price, 2),
        "optimal_demand": round(optimal_demand, 1),
        "optimal_profit": round(optimal_profit, 2),
    }


def price_sensitivity_curve(base_price: float, base_demand: float, elasticity: float,
                             unit_cost: float, n_points: int = 60) -> dict:
    prices   = np.linspace(unit_cost * 1.05, base_price * 2.2, n_points)
    demands  = [demand_at_price(p, base_price, base_demand, elasticity) for p in prices]
    revenues = [p * d for p, d in zip(prices, demands)]
    profits  = [(p - unit_cost) * d for p, d in zip(prices, demands)]
    return {
        "prices":   prices,
        "demands":  demands,
        "revenues": revenues,
        "profits":  profits,
    }
