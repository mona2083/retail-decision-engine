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

    # 変数の定義
    order_qty    = [model.NewIntVar(0, max_order_qty, f"order_{t}") for t in range(n_weeks)]
    stock        = [model.NewIntVar(0, max_stock * 2,  f"stock_{t}") for t in range(n_weeks)]
    order_flag   = [model.NewBoolVar(f"flag_{t}")                    for t in range(n_weeks)]
    shortage     = [model.NewIntVar(0, max_order_qty,  f"short_{t}") for t in range(n_weeks)]
    
    # 【追加】ソフト制約用のスラック変数（最低在庫を下回った量）
    below_min    = [model.NewIntVar(0, max_stock, f"below_min_{t}") for t in range(n_weeks)]

    # コストのスケーリング（ソルバーが扱いやすいよう整数化）
    SCALE = 100
    order_cost_int   = int(order_cost   * SCALE)
    holding_cost_int = int(holding_cost * SCALE)
    unit_cost_int    = int(unit_cost    * SCALE)
    
    # ペナルティ設定のトレードオフ
    # 欠品ペナルティ（最も高い） > 安全在庫割れペナルティ（中） > 保管コスト（低い）
    shortage_cost_int = int(unit_cost * 3.0 * SCALE)
    min_stock_penalty = int(unit_cost * 1.0 * SCALE)

    for t in range(n_weeks):
        arrival = order_qty[t - lead_time_weeks] if t >= lead_time_weeks else 0
        prev_stock = current_stock if t == 0 else stock[t - 1]
        
        # 1. 在庫バランス方程式（元のコードの両辺相殺バグを修正）
        # 「現在の在庫 - 欠品 = 前週の在庫 + 入荷 - 需要」として欠品を正しく許容
        model.Add(stock[t] - shortage[t] == prev_stock + arrival - demands[t])
        
        # 2. ハード制約：最大在庫水準（倉庫の物理的限界）
        model.Add(stock[t] <= max_stock)
        
        # 3. ソフト制約：最低在庫水準（安全在庫）
        # stock[t] + below_min[t] >= min_stock とすることで、
        # min_stockを下回った分だけ below_min[t] がプラスになりペナルティが課される
        model.Add(stock[t] + below_min[t] >= min_stock)

        # 4. 発注フラグの制約（発注量が1以上ならフラグを立てる）
        model.Add(order_qty[t] <= max_order_qty * order_flag[t])
        model.Add(order_qty[t] >= order_flag[t])

    # 目的関数（総コストの最小化）
    total_cost = (
        sum(order_flag[t] * order_cost_int + order_qty[t] * unit_cost_int for t in range(n_weeks)) +
        sum(stock[t] * holding_cost_int for t in range(n_weeks)) +
        sum(shortage[t] * shortage_cost_int for t in range(n_weeks)) +
        sum(below_min[t] * min_stock_penalty for t in range(n_weeks)) # 【追加】最低在庫割れのペナルティ
    )
    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # フォールバック処理（万が一解けなかった場合の単純計算）
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

    return {
        "status":     "ok",
        "order_qty":  [solver.Value(order_qty[t]) for t in range(n_weeks)],
        "stock":      [solver.Value(stock[t])     for t in range(n_weeks)],
        "shortage":   [solver.Value(shortage[t])  for t in range(n_weeks)],
        "below_min":  [solver.Value(below_min[t]) for t in range(n_weeks)],
        "total_cost": solver.ObjectiveValue() / SCALE,
        "is_optimal": status == cp_model.OPTIMAL,
    }


def simple_eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    return (2 * annual_demand * order_cost / holding_cost_per_unit) ** 0.5