import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import (
    generate_weekly_sales, 
    generate_daily_sales, 
    PRODUCTS, 
    SEASONAL_FACTORS,
    generate_daily_sales_with_price,
    aggregate_to_weekly              
)
from forecasting import fit_forecast, decompose_weekly_patterns
from pricing import demand_at_price, profit_at_price, find_optimal_price, price_sensitivity_curve
from inventory import run_inventory_optimization, simple_eoq
from tft_inference import load_tft_model, predict_dynamic_demand
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels.tsa.holtwinters.model")
warnings.filterwarnings("ignore", message=".*predict_dataloader.*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*Attribute 'loss'.*")
warnings.filterwarnings("ignore", message=".*Attribute 'logging_metrics'.*")

st.set_page_config(page_title="Retail Decision Engine", layout="wide")

def preprocess_for_tft(df: pd.DataFrame) -> pd.DataFrame:
    """TFTモデルの推論に必要なスキーマ（型とインデックス）を整える前処理"""
    # 1. 必須カラムの確認と補完（古いCSVアップロード対策）
    if "price" not in df.columns:
        # デモ以外でpriceがない場合は、ベース価格で仮埋めする（※実務では警告を出すべき箇所）
        df["price"] = df["product_id"].map(lambda x: PRODUCTS[x]["base_price"])
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "weekday" not in df.columns:
        df["weekday"] = df["date"].dt.day_name()

    # 2. ソート（データリーク防止のための最重要プロセス）
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    
    # 3. time_idxの生成
    df["time_idx"] = df.groupby("product_id").cumcount()
    
    # 4. 型の厳密なキャスト（Training-Serving Skew防止）
    df["product_id"] = df["product_id"].astype(str)
    df["weekday"] = df["weekday"].astype(str)
    df["month"] = df["month"].astype(str)
    df["is_weekend"] = df["is_weekend"].astype(str)
    df["sales"] = df["sales"].astype(float)
    df["price"] = df["price"].astype(float)
    
    return df

@st.cache_resource(show_spinner="Loading AI Model...")
def get_model():
    return load_tft_model("models/tft_best_model.ckpt")

tft_model = get_model()

PORTFOLIO_URL = "https://mona2083.github.io/portfolio-2026/index.html"

LANG = {
    "ja": {
        "title":           "🏪 Retail Decision Engine",
        "caption":         "需要予測 × ダイナミックプライシング × 発注最適化",
        "portfolio_btn":   "🔗 ポートフォリオを見る",
        "portfolio_label": "ポートフォリオ",
        "data_header":     "データ",
        "use_demo":        "デモデータを使う",
        "upload_csv":      "CSVをアップロード",
        "csv_format":      "CSV形式: date, product_id, sales",
        "generate":        "🎲 データ生成 & モデル実行",
        "seed":            "ランダムシード",
        "upload_label":    "週次売上CSVをアップロード",
        "loaded":          "✅ 読み込み完了",
        "section_forecast_pricing": "📈 需要予測 ＆ 💰 ダイナミックプライシング",
        "section_inventory":        "📦 発注最適化",
        "section_dashboard":        "🎯 総合ダッシュボード",
        "forecast_weeks":  "予測期間（週）",
        "history":         "実績",
        "fitted":          "フィット値",
        "forecast":        "予測",
        "ci":              "95%信頼区間",
        "mape":            "MAPE",
        "weekday_title":   "曜日別平均売上",
        "monthly_title":   "月別平均売上",
        "units":           "個",
        "price_slider":    "価格を設定（$）",
        "optimal_price":   "最適価格",
        "demand_label":    "推定需要（週間）",
        "revenue_label":   "推定売上（週間）",
        "profit_label":    "推定利益（週間）",
        "profit_curve":    "価格 vs 利益カーブ",
        "demand_curve":    "価格 vs 需要カーブ",
        "elasticity_info": "価格弾力性",
        "opt_price_line":  "最適価格",
        "cur_price_line":  "現在の価格",
        "current_stock":   "現在の在庫数（個）",
        "min_stock":       "最低在庫水準（個）",
        "max_stock":       "最大在庫水準（個）",
        "order_cost":      "1回の発注コスト（$）",
        "holding_cost":    "保管コスト（$／個／週）",
        "run_inv":         "🚀 発注計画を最適化",
        "inv_result":      "最適発注計画（今後8週間）",
        "week":            "週",
        "order":           "発注数",
        "stock_level":     "在庫水準",
        "demand_plan":     "予測需要",
        "total_order_cost":"総コスト（発注＋保管）",
        "eoq":             "EOQ（経済的発注量）",
        "dashboard_caption":"各商品の今週のアクション推奨を一覧表示",
        "action_price":    "💰 価格推奨",
        "action_demand":   "📈 需要予測",
        "action_inv":      "📦 発注推奨",
        "raise_price":     "値上げ推奨",
        "lower_price":     "値下げ推奨",
        "keep_price":      "現状維持",
        "order_now":       "今週発注",
        "no_order":        "発注不要",
        "vs_last_week":    "先週比",
        "profit_impact":   "利益インパクト",
        "run_inv_first":   "発注最適化を実行してください",
        "loading":         "モデル計算中...",
        "peak_month":      "ピーク月",
        "upload_btn":    "📂 アップロード & モデル実行",
        "csv_desc":      "必要なカラム：",
        "csv_col1":      "• date：週の開始日（例：2024-01-01）",
        "csv_col2":      "• product_id：商品ID（milk / bread / oj）",
        "csv_col3":      "• sales：週次売上数量（整数）",
        "csv_example":   "例）date,product_id,sales\n2024-01-01,milk,840\n2024-01-01,bread,630",
    },
    "en": {
        "title":           "🏪 Retail Decision Engine",
        "caption":         "Demand Forecasting × Dynamic Pricing × Inventory Optimization",
        "portfolio_btn":   "🔗 View Portfolio",
        "portfolio_label": "Portfolio",
        "data_header":     "Data",
        "use_demo":        "Use demo data",
        "upload_csv":      "Upload CSV",
        "csv_format":      "CSV format: date, product_id, sales",
        "generate":        "🎲 Generate & Run Models",
        "seed":            "Random Seed",
        "upload_label":    "Upload weekly sales CSV",
        "loaded":          "✅ Loaded",
        "section_forecast_pricing": "📈 Demand Forecast  &  💰 Dynamic Pricing",
        "section_inventory":        "📦 Inventory Optimization",
        "section_dashboard":        "🎯 Summary Dashboard",
        "forecast_weeks":  "Forecast Horizon (weeks)",
        "history":         "Actual",
        "fitted":          "Fitted",
        "forecast":        "Forecast",
        "ci":              "95% CI",
        "mape":            "MAPE",
        "weekday_title":   "Avg Sales by Day of Week",
        "monthly_title":   "Avg Sales by Month",
        "units":           "units",
        "price_slider":    "Set Price ($)",
        "optimal_price":   "Optimal Price",
        "demand_label":    "Est. Weekly Demand",
        "revenue_label":   "Est. Weekly Revenue",
        "profit_label":    "Est. Weekly Profit",
        "profit_curve":    "Price vs Profit",
        "demand_curve":    "Price vs Demand",
        "elasticity_info": "Price Elasticity",
        "opt_price_line":  "Optimal",
        "cur_price_line":  "Current",
        "current_stock":   "Current Stock (units)",
        "min_stock":       "Min Stock Level (units)",
        "max_stock":       "Max Stock Level (units)",
        "order_cost":      "Order Cost per Order ($)",
        "holding_cost":    "Holding Cost ($/unit/week)",
        "run_inv":         "🚀 Optimize Order Plan",
        "inv_result":      "Optimal Order Plan (Next 8 Weeks)",
        "week":            "Week",
        "order":           "Order Qty",
        "stock_level":     "Stock Level",
        "demand_plan":     "Forecast Demand",
        "total_order_cost":"Total Cost (Order + Holding)",
        "eoq":             "EOQ (Economic Order Quantity)",
        "dashboard_caption":"This week's recommended actions across all products",
        "action_price":    "💰 Pricing",
        "action_demand":   "📈 Demand",
        "action_inv":      "📦 Order",
        "raise_price":     "Raise Price",
        "lower_price":     "Lower Price",
        "keep_price":      "Keep Price",
        "order_now":       "Order Now",
        "no_order":        "No Order",
        "vs_last_week":    "vs Last Week",
        "profit_impact":   "Profit Impact",
        "run_inv_first":   "Run inventory optimization first",
        "loading":         "Computing...",
        "peak_month":      "Peak Month",
        "upload_btn":    "📂 Upload & Run Models",
        "csv_desc":      "Required columns:",
        "csv_col1":      "• date: week start date (e.g. 2024-01-01)",
        "csv_col2":      "• product_id: milk / bread / oj",
        "csv_col3":      "• sales: weekly sales quantity (integer)",
        "csv_example":   "e.g. date,product_id,sales\n2024-01-01,milk,840\n2024-01-01,bread,630",
    },
}

MONTH_NAMES_JA = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
MONTH_NAMES_EN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]



# ── サイドバー ────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.radio("🌐 Language / 言語", ["日本語", "English"], horizontal=True)
    lang = "ja" if lang_choice == "日本語" else "en"
    T = LANG[lang]

    st.link_button(T["portfolio_btn"], PORTFOLIO_URL, width="stretch")
    st.divider()
    st.header(T["data_header"])

    # アクセシビリティ警告を消すため、空文字ではなくダミーのラベルを設定
    data_mode = st.radio("Data Source", [T["use_demo"], T["upload_csv"]], label_visibility="collapsed")

    if T["use_demo"] in data_mode:
        seed = st.number_input(T["seed"], value=42, step=1)
        if st.button(T["generate"], width="stretch"):
            # TFT用の3年分・価格連動データを生成
            raw_daily = generate_daily_sales_with_price(1095, random_state=int(seed))
            st.session_state.daily_df = preprocess_for_tft(raw_daily)
            st.session_state.weekly_df = aggregate_to_weekly(raw_daily)
            st.cache_data.clear()
            st.rerun()
        
        if "weekly_df" not in st.session_state:
            # 初回ロード時も同様に処理
            raw_daily = generate_daily_sales_with_price(1095, random_state=42)
            st.session_state.daily_df = preprocess_for_tft(raw_daily)
            st.session_state.weekly_df = aggregate_to_weekly(raw_daily)
            
    else:
        # CSVアップロード処理（既存コードを維持しつつ、前処理を挟む）
        # ... (中略: CSV読み込みのUI部分) ...
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded, parse_dates=["date"])
                required = ["date", "product_id", "sales"]
                if all(c in df_up.columns for c in required):
                    st.session_state.weekly_df = df_up
                    # CSVの場合は過去のdailyジェネレータでダミー生成した後、前処理を通す
                    dummy_daily = generate_daily_sales(365, random_state=42)
                    st.session_state.daily_df = preprocess_for_tft(dummy_daily)
                    st.cache_data.clear()
                    st.success(T["loaded"])
                else:
                    st.error(f"Missing columns: {required}")
            except Exception as e:
                st.error(str(e))
        if "weekly_df" not in st.session_state:
            raw_daily = generate_daily_sales_with_price(1095, random_state=42)
            st.session_state.daily_df = preprocess_for_tft(raw_daily)
            st.session_state.weekly_df = aggregate_to_weekly(raw_daily)

weekly_df = st.session_state.weekly_df
daily_df  = st.session_state.daily_df

head_l, head_r = st.columns([0.78, 0.22], vertical_alignment="center")
with head_l:
    st.title(T["title"])
    st.caption(T["caption"])
with head_r:
    st.link_button(T["portfolio_label"], PORTFOLIO_URL, width="stretch")


def pname(pid):
    p = PRODUCTS[pid]
    return f"{p['icon']} {p['ja'] if lang == 'ja' else p['en']}"


def render_product_tab(pid: str):
    info       = PRODUCTS[pid]
    base_price = info["base_price"]
    unit_cost  = info["unit_cost"]
    elasticity = info["elasticity"]
    base_demand= info["base_demand"] * 7

    # ── 需要予測 ＆ プライシング ──────────────────────────────────
    st.subheader(T["section_forecast_pricing"])

    fc_weeks = st.slider(T["forecast_weeks"], 4, 12, 8, key=f"fcwk_{pid}")

    with st.spinner(T["loading"]):
        prod_weekly = weekly_df[weekly_df["product_id"] == pid].set_index("date")["sales"]
        prod_weekly = prod_weekly.asfreq("W-MON")
        fc_result   = fit_forecast(prod_weekly, fc_weeks)
        decomp      = decompose_weekly_patterns(daily_df, pid)

    col_fc, col_pr = st.columns(2)

    # 需要予測グラフ
    with col_fc:
        st.caption(f"MAPE: {fc_result['mape']:.1f}%")
        fig_fc = go.Figure()
        fig_fc.add_scatter(x=fc_result["history"].index, y=fc_result["history"].values,
                           name=T["history"], line=dict(color="#1a4a7a", width=2))
        fig_fc.add_scatter(x=fc_result["fitted"].index, y=fc_result["fitted"].values,
                           name=T["fitted"], line=dict(color="#b7d5c8", width=1.5, dash="dot"))
        fc_idx = fc_result["forecast"].index
        fig_fc.add_scatter(x=fc_idx, y=fc_result["forecast"].values,
                           name=T["forecast"], line=dict(color="#b5451b", width=2.5))
        fig_fc.add_scatter(
            x=list(fc_idx) + list(fc_idx[::-1]),
            y=list(fc_result["upper"].values) + list(fc_result["lower"].values[::-1]),
            fill="toself", fillcolor="rgba(181,69,27,0.12)",
            line=dict(width=0), name=T["ci"],
        )
        fig_fc.update_layout(height=320, margin=dict(t=20,b=20,l=20,r=20),
                             legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_fc, width="stretch")

        # 曜日・月別
        col_w, col_m = st.columns(2)
        month_names = MONTH_NAMES_JA if lang == "ja" else MONTH_NAMES_EN
        with col_w:
            fig_wd = go.Figure(go.Bar(
                x=list(decomp["weekday_avg"].index), y=decomp["weekday_avg"].values,
                marker_color=["#b5451b" if v == decomp["weekday_avg"].max()
                              else "#2d6a4f" if v == decomp["weekday_avg"].min()
                              else "#a8c4e0" for v in decomp["weekday_avg"].values],
            ))
            fig_wd.update_layout(title=T["weekday_title"], height=240,
                                 margin=dict(t=36,b=10,l=10,r=10))
            st.plotly_chart(fig_wd, width="stretch")
        with col_m:
            fig_mo = go.Figure(go.Bar(
                x=[month_names[i-1] for i in decomp["monthly_avg"].index],
                y=decomp["monthly_avg"].values,
                marker_color=["#b5451b" if v == decomp["monthly_avg"].max()
                              else "#2d6a4f" if v == decomp["monthly_avg"].min()
                              else "#a8c4e0" for v in decomp["monthly_avg"].values],
            ))
            fig_mo.update_layout(title=T["monthly_title"], height=240,
                                 margin=dict(t=36,b=10,l=10,r=10))
            st.plotly_chart(fig_mo, width="stretch")

    # プライシング
    with col_pr:
        opt = find_optimal_price(base_price, base_demand, elasticity, unit_cost)
        current_price = st.slider(
            T["price_slider"],
            min_value=float(round(unit_cost * 1.1, 2)),
            max_value=float(round(base_price * 2.2, 2)),
            value=float(base_price), step=0.05,
            key=f"price_{pid}",
        )
        
        # ── TFTモデルによる動的需要推論 ──
        with st.spinner(T["loading"]):
            daily_data = st.session_state.daily_df 
            
            # tft_inference.py の関数を呼び出し、選択された価格での未来需要を予測
            weekly_forecast = predict_dynamic_demand(
                model=tft_model,
                daily_df=daily_data,
                product_id=pid,
                planned_price=current_price
            )
            # 未来8週間の予測需要の平均を「今週の推定需要」として採用
            demand_now = weekly_forecast["forecast_sales"].mean()

        profit_now  = (current_price - unit_cost) * demand_now
        opt_profit  = opt["optimal_profit"]

        # KPIメトリクス
        m1, m2, m3 = st.columns(3)
        m1.metric(T["demand_label"],  f"{demand_now:.0f} {T['units']}")
        m2.metric(T["revenue_label"], f"${current_price * demand_now:,.2f}")
        m3.metric(T["profit_label"],  f"${profit_now:,.2f}",
                  delta=f"${profit_now - opt_profit:+.2f} vs opt",
                  delta_color="inverse" if profit_now < opt_profit * 0.98 else "off")

        st.caption(f"✨ {T['optimal_price']}: **${opt['optimal_price']:.2f}** — {T['elasticity_info']}: {elasticity}")

        # グラフ描画（ハイブリッド方式：理論カーブ ＋ AI推論マーカー）
        curve = price_sensitivity_curve(base_price, base_demand, elasticity, unit_cost)
        fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=[T["profit_curve"], T["demand_curve"]])
        
        # 背景の理論カーブ
        fig_p.add_scatter(x=curve["prices"], y=curve["profits"],
                          line=dict(color="#2d6a4f", width=2, dash="dot"), row=1, col=1, name="理論利益")
        fig_p.add_scatter(x=curve["prices"], y=curve["demands"],
                          line=dict(color="#1a4a7a", width=2, dash="dot"), row=2, col=1, name="理論需要")
        
        # TFTモデルの予測地点（現在のスライダー値）
        fig_p.add_scatter(x=[current_price], y=[profit_now], 
                          mode="markers", marker=dict(color="#b5451b", size=14, symbol="star"), 
                          row=1, col=1, name="AI予測利益")
        fig_p.add_scatter(x=[current_price], y=[demand_now], 
                          mode="markers", marker=dict(color="#b5451b", size=14, symbol="star"), 
                          row=2, col=1, name="AI予測需要")

        for row in [1, 2]:
            fig_p.add_vline(x=opt["optimal_price"], line_dash="dash", line_color="#b5451b",
                            annotation_text=T["opt_price_line"], row=row, col=1)
            fig_p.add_vline(x=current_price, line_dash="dot", line_color="#888",
                            annotation_text=T["cur_price_line"], row=row, col=1)
            
        fig_p.update_layout(height=560, margin=dict(t=40,b=20,l=20,r=20), showlegend=False)
        st.plotly_chart(fig_p, width="stretch")

    st.divider()

    # ── 発注最適化 ────────────────────────────────────────────────
    st.subheader(T["section_inventory"])

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        current_stock = st.number_input(T["current_stock"], 0, 2000, 150, step=10, key=f"cs_{pid}")
        min_stock     = st.number_input(T["min_stock"],     0, 500,  50,  step=10, key=f"ms_{pid}")
    with col_i2:
        max_stock     = st.number_input(T["max_stock"],     100, 3000, 600, step=50, key=f"mx_{pid}")
        order_cost    = st.number_input(T["order_cost"],    1.0, 500.0, 50.0, step=5.0, key=f"oc_{pid}")
    with col_i3:
        holding_cost  = st.number_input(T["holding_cost"], 0.01, 5.0, 0.10, step=0.01, key=f"hc_{pid}")

    if st.button(T["run_inv"], type="primary", width="stretch", key=f"runinv_{pid}"):
        with st.spinner(T["loading"]):
            # 上のプライシングセクションでTFTが算出した「設定価格における未来8週間の需要予測」をそのまま配列化
            forecast_8w = weekly_forecast["forecast_sales"].tolist()
            
            # OR-Toolsによる発注最適化ソルバーの実行
            inv_result  = run_inventory_optimization(
                forecast_demand=forecast_8w,
                current_stock=int(current_stock),
                min_stock=int(min_stock),
                max_stock=int(max_stock),
                order_cost=float(order_cost),
                holding_cost=float(holding_cost),
                unit_cost=float(unit_cost),
            )
            
            # 結果をセッションステートに保存（グラフ描画用）
            st.session_state[f"inv_{pid}"] = {
                "result": inv_result, 
                "forecast": forecast_8w,
                "order_cost": order_cost, 
                "holding_cost": holding_cost,
            }

    if f"inv_{pid}" in st.session_state:
        inv      = st.session_state[f"inv_{pid}"]["result"]
        fc_vals  = st.session_state[f"inv_{pid}"]["forecast"]
        n        = len(inv["order_qty"])
        wk_lbl   = [f"W+{i+1}" for i in range(n)]

        col_m1, col_m2 = st.columns(2)
        col_m1.metric(T["total_order_cost"], f"${inv['total_cost']:,.2f}")
        eoq_val = simple_eoq(
            annual_demand=sum(fc_vals) * (52/8),
            order_cost=st.session_state[f"inv_{pid}"]["order_cost"],
            holding_cost_per_unit=st.session_state[f"inv_{pid}"]["holding_cost"] * 52,
        )
        col_m2.metric(T["eoq"], f"{eoq_val:.0f} {T['units']}")

        fig_inv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=[T["stock_level"], T["order"]])
        fig_inv.add_bar(x=wk_lbl, y=inv["stock"], name=T["stock_level"],
                        marker_color="#2d6a4f", opacity=0.75, row=1, col=1)
        fig_inv.add_scatter(x=wk_lbl, y=[min_stock]*n,
                            line=dict(color="#c0392b", dash="dash"), row=1, col=1,
                            name=("最低在庫" if lang == "ja" else "Min Stock"))
        fig_inv.add_bar(x=wk_lbl, y=inv["order_qty"], name=T["order"],
                        marker_color="#b5451b", opacity=0.85, row=2, col=1)
        fig_inv.add_scatter(x=wk_lbl, y=[int(d) for d in fc_vals],
                            line=dict(color="#1a4a7a", width=2), name=T["demand_plan"], row=2, col=1)
        fig_inv.update_layout(height=400, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_inv, width="stretch")

        inv_df = pd.DataFrame({
            T["week"]:        wk_lbl,
            T["demand_plan"]: [int(d) for d in fc_vals],
            T["order"]:       inv["order_qty"],
            T["stock_level"]: inv["stock"],
        })
        st.dataframe(inv_df, width="stretch", hide_index=True)


# ── プロダクトタブ ────────────────────────────────────────────────
tabs = st.tabs([pname(pid) for pid in PRODUCTS])
for tab, pid in zip(tabs, PRODUCTS):
    with tab:
        render_product_tab(pid)

st.divider()

# ── 総合ダッシュボード ────────────────────────────────────────────
st.header(T["section_dashboard"])
st.caption(T["dashboard_caption"])

dash_cols = st.columns(len(PRODUCTS))
for col, pid in zip(dash_cols, PRODUCTS):
    info  = PRODUCTS[pid]
    with col:
        st.markdown(f"### {pname(pid)}")

        # ユーザーが各タブで設定している「現在の価格」を取得（未設定ならベース価格）
        current_p = st.session_state.get(f"price_{pid}", info["base_price"])

        # ── 1. 需要（TFTモデルによる動的予測へ完全移行） ──
        with st.spinner(f"Predicting {pid}..."):
            weekly_fc = predict_dynamic_demand(
                model=tft_model,
                daily_df=st.session_state.daily_df,
                product_id=pid,
                planned_price=current_p
            )
        # TFTが予測した次週（W+1）の需要を抽出
        next_d = weekly_fc["forecast_sales"].iloc[0]
        
        # 比較用の実績データ（先週の売上）を取得
        prod_w = st.session_state.weekly_df[st.session_state.weekly_df["product_id"] == pid].set_index("date")["sales"].asfreq("W-MON")
        last_d = prod_w.values[-1]
        
        delta  = (next_d - last_d) / last_d * 100
        month_names = MONTH_NAMES_JA if lang == "ja" else MONTH_NAMES_EN
        peak_m = SEASONAL_FACTORS[pid].index(max(SEASONAL_FACTORS[pid]))

        st.metric(T["action_demand"],
                  f"{next_d:.0f} {T['units']}",
                  delta=f"{delta:+.1f}% {T['vs_last_week']}",
                  delta_color="normal" if delta >= 0 else "inverse")
        st.caption(f"{T['peak_month']}: {month_names[peak_m]}")

        # ── 2. 価格（現在価格と最適価格のギャップ分析） ──
        opt_d = find_optimal_price(info["base_price"], info["base_demand"]*7,
                                   info["elasticity"], info["unit_cost"])
        
        # 「現在設定している価格」と「理論最適価格」の差分を計算
        price_diff = opt_d["optimal_price"] - current_p
        
        # ギャップを埋めた場合に得られる推定利益のアップサイド
        profit_imp = opt_d["optimal_profit"] - profit_at_price(
            current_p, info["base_price"], info["base_demand"]*7,
            info["elasticity"], info["unit_cost"])

        if abs(price_diff) < 0.05:
            lbl = T["keep_price"]; dc = "off"
        elif price_diff > 0:
            lbl = T["raise_price"]; dc = "normal"
        else:
            lbl = T["lower_price"]; dc = "inverse"

        st.metric(T["action_price"],
                  f"${opt_d['optimal_price']:.2f}",
                  delta=f"{price_diff:+.2f} ({lbl})",
                  delta_color=dc)
        st.caption(f"{T['profit_impact']}: ${profit_imp:+.2f}/{'週' if lang=='ja' else 'week'}")

        # ── 3. 発注（連携済みソルバー結果の呼び出し） ──
        inv_key = f"inv_{pid}"
        if inv_key in st.session_state:
            order_w1 = st.session_state[inv_key]["result"]["order_qty"][0]
            if order_w1 > 0:
                st.metric(T["action_inv"], f"{order_w1} {T['units']}", delta=T["order_now"], delta_color="normal")
            else:
                st.metric(T["action_inv"], "—", delta=T["no_order"], delta_color="off")
        else:
            st.caption(f"📦 {T['run_inv_first']}")