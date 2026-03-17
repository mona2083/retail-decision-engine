import numpy as np
from ortools.sat.python import cp_model


def run_inventory_optimization(
    forecast_demand: list[float],
    current_stock:   int,
    min_stock:       int,
    max_stock:       int,
    order_cost:      float,
    holding_cost:    float,
    unit_cost:       float,
    lead_time_weeks: int = 1,
    max_order_qty:   int = 500,
) -> dict:
    n_weeks = len(forecast_demand)
    demands = [max(int(d), 0) for d in forecast_demand]

    model = cp_model.CpModel()

    order_qty    = [model.NewIntVar(0, max_order_qty, f"order_{t}") for t in range(n_weeks)]
    stock        = [model.NewIntVar(0, max_stock * 2,  f"stock_{t}") for t in range(n_weeks)]
    order_flag   = [model.NewBoolVar(f"flag_{t}")                    for t in range(n_weeks)]
    shortage     = [model.NewIntVar(0, max_order_qty,  f"short_{t}") for t in range(n_weeks)]

    SCALE = 100
    order_cost_int   = int(order_cost   * SCALE)
    holding_cost_int = int(holding_cost * SCALE)
    unit_cost_int    = int(unit_cost    * SCALE)
    shortage_cost_int = int(unit_cost * 3 * SCALE)

    for t in range(n_weeks):
        if t == 0:
            arrival = order_qty[t - lead_time_weeks] if t >= lead_time_weeks else 0
            model.Add(stock[t] + shortage[t] == current_stock + arrival - demands[t] + shortage[t])
            model.Add(stock[t] >= 0)
        else:
            arrival = order_qty[t - lead_time_weeks] if t >= lead_time_weeks else 0
            model.Add(stock[t] + shortage[t] == stock[t - 1] + arrival - demands[t])
            model.Add(stock[t] >= 0)

        model.Add(stock[t] <= max_stock)
        model.Add(order_qty[t] <= max_order_qty * order_flag[t])
        model.Add(order_qty[t] >= order_flag[t])

    total_cost = (
        sum(order_flag[t] * order_cost_int + order_qty[t] * unit_cost_int for t in range(n_weeks)) +
        sum(stock[t] * holding_cost_int                                    for t in range(n_weeks)) +
        sum(shortage[t] * shortage_cost_int                                for t in range(n_weeks))
    )
    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        order_qty_simple = [max(int(d) - current_stock + min_stock, 0) for d in demands]
        stock_simple = []
        s = current_stock
        for t, (d, o) in enumerate(zip(demands, order_qty_simple)):
            s = max(s + (o if t >= lead_time_weeks else 0) - d, 0)
            stock_simple.append(s)
        return {
            "status":    "fallback",
            "order_qty": order_qty_simple,
            "stock":     stock_simple,
            "total_cost": sum(order_qty_simple) * unit_cost,
        }

    order_qty_vals = [solver.Value(order_qty[t]) for t in range(n_weeks)]
    stock_vals     = [solver.Value(stock[t])     for t in range(n_weeks)]
    shortage_vals  = [solver.Value(shortage[t])  for t in range(n_weeks)]

    return {
        "status":     "ok",
        "order_qty":  order_qty_vals,
        "stock":      stock_vals,
        "shortage":   shortage_vals,
        "total_cost": solver.ObjectiveValue() / SCALE,
        "is_optimal": status == cp_model.OPTIMAL,
    }


def simple_eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    return (2 * annual_demand * order_cost / holding_cost_per_unit) ** 0.5
