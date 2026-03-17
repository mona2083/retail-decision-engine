import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_generator import generate_weekly_sales, generate_daily_sales, PRODUCTS, SEASONAL_FACTORS
from forecasting import fit_forecast, decompose_weekly_patterns
from pricing import demand_at_price, profit_at_price, find_optimal_price, price_sensitivity_curve
from inventory import run_inventory_optimization, simple_eoq

st.set_page_config(page_title="Retail Intelligence Dashboard", layout="wide")

LANG = {
    "ja": {
        "title":          "🏪 Retail Intelligence Dashboard",
        "caption":        "需要予測 × ダイナミックプライシング × 発注最適化",
        "tab1":           "📈 需要予測",
        "tab2":           "💰 ダイナミックプライシング",
        "tab3":           "📦 発注最適化",
        "tab4":           "🎯 総合ダッシュボード",
        "product":        "商品を選択",
        "forecast_weeks": "予測期間（週）",
        "history":        "実績",
        "fitted":         "フィット値",
        "forecast":       "予測",
        "ci":             "95%信頼区間",
        "mape":           "MAPE（平均絶対誤差率）",
        "weekday_title":  "曜日別平均売上",
        "monthly_title":  "月別平均売上",
        "units":          "個",
        "price_slider":   "価格を設定（$）",
        "current_price":  "現在の価格",
        "optimal_price":  "最適価格",
        "demand_label":   "推定需要（週間）",
        "revenue_label":  "推定売上（週間）",
        "profit_label":   "推定利益（週間）",
        "profit_curve":   "価格 vs 利益カーブ",
        "demand_curve":   "価格 vs 需要カーブ",
        "elasticity_info":"価格弾力性",
        "opt_price_line": "最適価格",
        "cur_price_line": "現在の価格",
        "current_stock":  "現在の在庫数（個）",
        "min_stock":      "最低在庫水準（個）",
        "max_stock":      "最大在庫水準（個）",
        "order_cost":     "1回の発注コスト（$）",
        "holding_cost":   "保管コスト（$／個／週）",
        "run_inv":        "🚀 発注計画を最適化",
        "inv_result":     "最適発注計画（今後8週間）",
        "week":           "週",
        "order":          "発注数",
        "stock_level":    "在庫水準",
        "demand_plan":    "予測需要",
        "total_order_cost":"総コスト（発注＋保管）",
        "eoq":            "EOQ（経済的発注量）",
        "dashboard_title":"今週のアクションサマリー",
        "action_price":   "💰 価格推奨",
        "action_demand":  "📈 需要予測",
        "action_inv":     "📦 発注推奨",
        "raise_price":    "値上げ推奨",
        "lower_price":    "値下げ推奨",
        "keep_price":     "現状維持",
        "order_now":      "今週発注すべき",
        "no_order":       "今週発注不要",
        "vs_last_week":   "先週比",
        "profit_impact":  "利益インパクト",
        "integrated_title":"3つの最適化を統合した推奨アクション",
        "loading":        "モデル学習中...",
    },
    "en": {
        "title":          "🏪 Retail Intelligence Dashboard",
        "caption":        "Demand Forecasting × Dynamic Pricing × Inventory Optimization",
        "tab1":           "📈 Demand Forecast",
        "tab2":           "💰 Dynamic Pricing",
        "tab3":           "📦 Inventory Optimization",
        "tab4":           "🎯 Summary Dashboard",
        "product":        "Select Product",
        "forecast_weeks": "Forecast Horizon (weeks)",
        "history":        "Actual",
        "fitted":         "Fitted",
        "forecast":       "Forecast",
        "ci":             "95% Confidence Interval",
        "mape":           "MAPE (Mean Absolute % Error)",
        "weekday_title":  "Average Sales by Day of Week",
        "monthly_title":  "Average Sales by Month",
        "units":          "units",
        "price_slider":   "Set Price ($)",
        "current_price":  "Current Price",
        "optimal_price":  "Optimal Price",
        "demand_label":   "Est. Weekly Demand",
        "revenue_label":  "Est. Weekly Revenue",
        "profit_label":   "Est. Weekly Profit",
        "profit_curve":   "Price vs Profit Curve",
        "demand_curve":   "Price vs Demand Curve",
        "elasticity_info":"Price Elasticity",
        "opt_price_line": "Optimal Price",
        "cur_price_line": "Current Price",
        "current_stock":  "Current Stock (units)",
        "min_stock":      "Min Stock Level (units)",
        "max_stock":      "Max Stock Level (units)",
        "order_cost":     "Order Cost per Order ($)",
        "holding_cost":   "Holding Cost ($/unit/week)",
        "run_inv":        "🚀 Optimize Order Plan",
        "inv_result":     "Optimal Order Plan (Next 8 Weeks)",
        "week":           "Week",
        "order":          "Order Qty",
        "stock_level":    "Stock Level",
        "demand_plan":    "Forecast Demand",
        "total_order_cost":"Total Cost (Order + Holding)",
        "eoq":            "EOQ (Economic Order Quantity)",
        "dashboard_title":"This Week's Action Summary",
        "action_price":   "💰 Pricing Recommendation",
        "action_demand":  "📈 Demand Forecast",
        "action_inv":     "📦 Order Recommendation",
        "raise_price":    "Raise Price",
        "lower_price":    "Lower Price",
        "keep_price":     "Keep Current Price",
        "order_now":      "Order This Week",
        "no_order":       "No Order Needed",
        "vs_last_week":   "vs Last Week",
        "profit_impact":  "Profit Impact",
        "integrated_title":"Integrated Recommendations from 3 Optimization Modules",
        "loading":        "Training model...",
    },
}

MONTH_NAMES_JA = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
MONTH_NAMES_EN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── 言語設定 ──────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.radio("🌐 Language / 言語", ["日本語", "English"], horizontal=True)
    lang = "ja" if lang_choice == "日本語" else "en"

T = LANG[lang]
st.title(T["title"])
st.caption(T["caption"])

# ── データ生成（キャッシュ）────────────────────────────────────────
@st.cache_data
def get_data():
    return generate_weekly_sales(104), generate_daily_sales(365)

weekly_df, daily_df = get_data()

# ── 商品名マッピング ──────────────────────────────────────────────
def pname(pid):
    p = PRODUCTS[pid]
    return f"{p['icon']} {p['ja'] if lang == 'ja' else p['en']}"

product_options = {pname(pid): pid for pid in PRODUCTS}

tab1, tab2, tab3, tab4 = st.tabs([T["tab1"], T["tab2"], T["tab3"], T["tab4"]])

# ══════════════════════════════════════════════════════════════════
# TAB 1: 需要予測
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_sel, col_wk = st.columns([2, 1])
    with col_sel:
        sel_label1  = st.selectbox(T["product"], list(product_options.keys()), key="prod1")
        product_id1 = product_options[sel_label1]
    with col_wk:
        fc_weeks = st.slider(T["forecast_weeks"], 4, 12, 8, key="fcwk")

    with st.spinner(T["loading"]):
        prod_weekly = weekly_df[weekly_df["product_id"] == product_id1].set_index("date")["sales"]
        fc_result   = fit_forecast(prod_weekly, fc_weeks)
        decomp      = decompose_weekly_patterns(daily_df, product_id1)

    # 予測精度
    st.metric(T["mape"], f"{fc_result['mape']:.1f}%")

    # ── 需要予測グラフ ──────────────────────────────────────────────
    fig_fc = go.Figure()
    fig_fc.add_scatter(
        x=fc_result["history"].index, y=fc_result["history"].values,
        name=T["history"], line=dict(color="#1a4a7a", width=2),
    )
    fig_fc.add_scatter(
        x=fc_result["fitted"].index, y=fc_result["fitted"].values,
        name=T["fitted"], line=dict(color="#b7d5c8", width=1.5, dash="dot"),
    )
    fc_idx = fc_result["forecast"].index
    fig_fc.add_scatter(
        x=fc_idx, y=fc_result["forecast"].values,
        name=T["forecast"], line=dict(color="#b5451b", width=2.5),
    )
    fig_fc.add_scatter(
        x=list(fc_idx) + list(fc_idx[::-1]),
        y=list(fc_result["upper"].values) + list(fc_result["lower"].values[::-1]),
        fill="toself", fillcolor="rgba(181,69,27,0.12)",
        line=dict(width=0), name=T["ci"], showlegend=True,
    )
    fig_fc.update_layout(height=380, margin=dict(t=20,b=20,l=20,r=20),
                         legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── 曜日・月別パターン ──────────────────────────────────────────
    col_w, col_m = st.columns(2)
    with col_w:
        fig_wd = go.Figure(go.Bar(
            x=list(decomp["weekday_avg"].index),
            y=decomp["weekday_avg"].values,
            marker_color=["#b5451b" if v == decomp["weekday_avg"].max()
                          else "#2d6a4f" if v == decomp["weekday_avg"].min()
                          else "#a8c4e0" for v in decomp["weekday_avg"].values],
        ))
        fig_wd.update_layout(title=T["weekday_title"], height=300,
                             margin=dict(t=40,b=20,l=20,r=20),
                             yaxis_title=T["units"])
        st.plotly_chart(fig_wd, use_container_width=True)

    with col_m:
        month_names = MONTH_NAMES_JA if lang == "ja" else MONTH_NAMES_EN
        fig_mo = go.Figure(go.Bar(
            x=[month_names[i-1] for i in decomp["monthly_avg"].index],
            y=decomp["monthly_avg"].values,
            marker_color=["#b5451b" if v == decomp["monthly_avg"].max()
                          else "#2d6a4f" if v == decomp["monthly_avg"].min()
                          else "#a8c4e0" for v in decomp["monthly_avg"].values],
        ))
        fig_mo.update_layout(title=T["monthly_title"], height=300,
                             margin=dict(t=40,b=20,l=20,r=20),
                             yaxis_title=T["units"])
        st.plotly_chart(fig_mo, use_container_width=True)

    # 今後の予測テーブル
    fc_df = pd.DataFrame({
        ("週" if lang == "ja" else "Week"):           [f"W+{i+1}" for i in range(fc_weeks)],
        T["forecast"]:                                fc_result["forecast"].values.round(0).astype(int),
        ("下限 (95%)" if lang == "ja" else "Lower"): fc_result["lower"].values.round(0).astype(int),
        ("上限 (95%)" if lang == "ja" else "Upper"): fc_result["upper"].values.round(0).astype(int),
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: ダイナミックプライシング
# ══════════════════════════════════════════════════════════════════
with tab2:
    sel_label2  = st.selectbox(T["product"], list(product_options.keys()), key="prod2")
    product_id2 = product_options[sel_label2]
    info2       = PRODUCTS[product_id2]

    base_price2  = info2["base_price"]
    unit_cost2   = info2["unit_cost"]
    elasticity2  = info2["elasticity"]
    base_demand2 = info2["base_demand"] * 7

    opt2 = find_optimal_price(base_price2, base_demand2, elasticity2, unit_cost2)

    col_sl, col_curve = st.columns([1, 2])
    with col_sl:
        current_price2 = st.slider(
            T["price_slider"],
            min_value=float(round(unit_cost2 * 1.1, 2)),
            max_value=float(round(base_price2 * 2.2, 2)),
            value=float(base_price2),
            step=0.05,
            key="price_sl",
        )

        demand_now  = demand_at_price(current_price2, base_price2, base_demand2, elasticity2)
        revenue_now = current_price2 * demand_now
        profit_now  = (current_price2 - unit_cost2) * demand_now
        opt_profit  = opt2["optimal_profit"]

        st.metric(T["demand_label"],  f"{demand_now:.0f} {T['units']}")
        st.metric(T["revenue_label"], f"${revenue_now:,.2f}")
        st.metric(T["profit_label"],  f"${profit_now:,.2f}",
                  delta=f"${profit_now - opt_profit:+.2f} vs optimal",
                  delta_color="inverse" if profit_now < opt_profit else "off")

        st.divider()
        st.subheader(T["optimal_price"])
        st.metric("", f"${opt2['optimal_price']:.2f}")
        st.caption(
            f"需要：{opt2['optimal_demand']:.0f}{T['units']}　利益：${opt2['optimal_profit']:,.2f}"
            if lang == "ja" else
            f"Demand: {opt2['optimal_demand']:.0f} {T['units']}　Profit: ${opt2['optimal_profit']:,.2f}"
        )
        st.caption(f"{T['elasticity_info']}: {elasticity2}")

    with col_curve:
        curve2 = price_sensitivity_curve(base_price2, base_demand2, elasticity2, unit_cost2)

        fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=[T["profit_curve"], T["demand_curve"]])

        fig_price.add_scatter(x=curve2["prices"], y=curve2["profits"],
                              name=T["profit_label"], line=dict(color="#2d6a4f", width=2.5),
                              row=1, col=1)
        fig_price.add_scatter(x=curve2["prices"], y=curve2["demands"],
                              name=T["demand_label"], line=dict(color="#1a4a7a", width=2.5),
                              row=2, col=1)

        for row in [1, 2]:
            fig_price.add_vline(x=opt2["optimal_price"],  line_dash="dash",
                                line_color="#b5451b", row=row, col=1,
                                annotation_text=T["opt_price_line"])
            fig_price.add_vline(x=current_price2, line_dash="dot",
                                line_color="#888", row=row, col=1,
                                annotation_text=T["cur_price_line"])

        fig_price.update_layout(height=420, margin=dict(t=40,b=20,l=20,r=20),
                                showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3: 発注最適化
# ══════════════════════════════════════════════════════════════════
with tab3:
    sel_label3  = st.selectbox(T["product"], list(product_options.keys()), key="prod3")
    product_id3 = product_options[sel_label3]
    info3       = PRODUCTS[product_id3]

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        current_stock = st.number_input(T["current_stock"], 0, 2000, 150, step=10)
        min_stock     = st.number_input(T["min_stock"],     0, 500,  50,  step=10)
    with col_p2:
        max_stock     = st.number_input(T["max_stock"],     100, 3000, 600, step=50)
        order_cost    = st.number_input(T["order_cost"],    1.0, 500.0, 50.0, step=5.0)
    with col_p3:
        holding_cost  = st.number_input(T["holding_cost"],  0.01, 5.0, 0.10, step=0.01)

    run_inv = st.button(T["run_inv"], type="primary", use_container_width=True)

    if run_inv:
        with st.spinner(T["loading"]):
            prod_weekly3 = weekly_df[weekly_df["product_id"] == product_id3].set_index("date")["sales"]
            fc3          = fit_forecast(prod_weekly3, 8)
            forecast_8w  = fc3["forecast"].values.tolist()

            inv_result = run_inventory_optimization(
                forecast_demand=forecast_8w,
                current_stock=int(current_stock),
                min_stock=int(min_stock),
                max_stock=int(max_stock),
                order_cost=float(order_cost),
                holding_cost=float(holding_cost),
                unit_cost=float(info3["unit_cost"]),
            )
            st.session_state[f"inv_{product_id3}"] = {
                "result": inv_result, "forecast": forecast_8w,
                "order_cost": order_cost, "holding_cost": holding_cost,
            }

    key3 = f"inv_{product_id3}"
    if key3 in st.session_state:
        inv = st.session_state[key3]["result"]
        fc_vals = st.session_state[key3]["forecast"]

        st.subheader(T["inv_result"])
        st.metric(T["total_order_cost"], f"${inv['total_cost']:,.2f}")

        eoq_val = simple_eoq(
            annual_demand=sum(fc_vals) * (52 / 8),
            order_cost=st.session_state[key3]["order_cost"],
            holding_cost_per_unit=st.session_state[key3]["holding_cost"] * 52,
        )
        st.caption(f"{T['eoq']}: {eoq_val:.0f} {T['units']}")

        n = len(inv["order_qty"])
        week_labels = [f"W+{i+1}" for i in range(n)]

        fig_inv = make_subplots(rows=2, cols=1,
                                subplot_titles=[T["inv_result"], T["demand_plan"]],
                                shared_xaxes=True)

        fig_inv.add_bar(x=week_labels, y=inv["stock"], name=T["stock_level"],
                        marker_color="#2d6a4f", opacity=0.7, row=1, col=1)
        fig_inv.add_scatter(x=week_labels, y=[min_stock]*n,
                            name=("最低在庫" if lang == "ja" else "Min Stock"),
                            line=dict(color="#c0392b", dash="dash"), row=1, col=1)
        fig_inv.add_bar(x=week_labels, y=inv["order_qty"], name=T["order"],
                        marker_color="#b5451b", opacity=0.85, row=2, col=1)
        fig_inv.add_scatter(x=week_labels, y=[int(d) for d in fc_vals],
                            name=T["demand_plan"], line=dict(color="#1a4a7a", width=2),
                            row=2, col=1)

        fig_inv.update_layout(height=440, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_inv, use_container_width=True)

        inv_df = pd.DataFrame({
            T["week"]:        week_labels,
            T["demand_plan"]: [int(d) for d in fc_vals],
            T["order"]:       inv["order_qty"],
            T["stock_level"]: inv["stock"],
        })
        st.dataframe(inv_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4: 総合ダッシュボード
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(T["dashboard_title"])
    st.caption(T["integrated_title"])

    for pid, info in PRODUCTS.items():
        pn = pname(pid)

        prod_weekly_d = weekly_df[weekly_df["product_id"] == pid].set_index("date")["sales"]
        fc_d          = fit_forecast(prod_weekly_d, 4)
        next_demand   = fc_d["forecast"].values[0]
        last_demand   = prod_weekly_d.values[-1]
        demand_delta  = (next_demand - last_demand) / last_demand * 100

        opt_d = find_optimal_price(info["base_price"], info["base_demand"] * 7,
                                   info["elasticity"], info["unit_cost"])
        price_diff    = opt_d["optimal_price"] - info["base_price"]
        profit_impact = opt_d["optimal_profit"] - profit_at_price(
            info["base_price"], info["base_price"], info["base_demand"] * 7,
            info["elasticity"], info["unit_cost"]
        )

        inv_key = f"inv_{pid}"
        has_inv = inv_key in st.session_state

        with st.expander(f"**{pn}**", expanded=True):
            c1, c2, c3 = st.columns(3)

            # 価格推奨
            with c1:
                st.markdown(f"#### {T['action_price']}")
                if abs(price_diff) < 0.05:
                    label = T["keep_price"]
                    delta_color = "off"
                elif price_diff > 0:
                    label = T["raise_price"]
                    delta_color = "normal"
                else:
                    label = T["lower_price"]
                    delta_color = "inverse"
                st.metric(
                    label,
                    f"${opt_d['optimal_price']:.2f}",
                    delta=f"{price_diff:+.2f} vs ${info['base_price']:.2f}",
                    delta_color=delta_color,
                )
                st.caption(f"{T['profit_impact']}: ${profit_impact:+.2f}/{'週' if lang == 'ja' else 'week'}")

            # 需要予測
            with c2:
                st.markdown(f"#### {T['action_demand']}")
                st.metric(
                    ("来週予測" if lang == "ja" else "Next Week Forecast"),
                    f"{next_demand:.0f} {T['units']}",
                    delta=f"{demand_delta:+.1f}% {T['vs_last_week']}",
                    delta_color="normal" if demand_delta >= 0 else "inverse",
                )
                peak_month  = SEASONAL_FACTORS[pid].index(max(SEASONAL_FACTORS[pid])) + 1
                month_names = MONTH_NAMES_JA if lang == "ja" else MONTH_NAMES_EN
                st.caption(
                    f"ピーク月：{month_names[peak_month-1]}" if lang == "ja"
                    else f"Peak month: {month_names[peak_month-1]}"
                )

            # 発注推奨
            with c3:
                st.markdown(f"#### {T['action_inv']}")
                if has_inv:
                    inv_data = st.session_state[inv_key]["result"]
                    order_w1 = inv_data["order_qty"][0]
                    if order_w1 > 0:
                        st.metric(T["order_now"], f"{order_w1} {T['units']}", delta_color="normal")
                    else:
                        st.metric(T["no_order"], "—")
                else:
                    st.info(
                        "発注最適化タブで実行してください" if lang == "ja"
                        else "Run optimization in the Inventory tab"
                    )

        st.divider()
