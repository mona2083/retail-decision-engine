# 🏪 Retail Decision Engine

> Demand Forecasting × Dynamic Pricing × Inventory Optimization — all in one dashboard.

A Streamlit app that combines three AI/optimization models to give grocery and retail managers a weekly decision brief: what price to set, how much demand to expect, and when to reorder.

---

## Live Demo

🔗 [Open App](https://retail-decision-engine-4h8ohlz3u2zz5tvzkubhh2.streamlit.app/) 

---

## Features

### 📈 Demand Forecasting
- Holt-Winters exponential smoothing trained on 104 weeks of sales history
- Seasonal decomposition (weekly + monthly patterns)
- 8-week forecast with 95% confidence intervals
- MAPE (Mean Absolute Percentage Error) displayed for model transparency

### 💰 Dynamic Pricing
- Price elasticity model estimates demand response to price changes
- SciPy optimizer finds the profit-maximizing price in real time
- Interactive price slider — move it and watch demand, revenue, and profit update instantly
- Price vs. profit curve and price vs. demand curve visualized side by side

### 📦 Inventory Optimization
- OR-Tools CP-SAT solver plans optimal order quantities for the next 8 weeks
- Hard constraints: minimum safety stock, maximum warehouse capacity, lead time
- Soft objective: minimize total cost (order fixed cost + per-unit cost + holding cost)
- EOQ (Economic Order Quantity) benchmark from classical inventory theory

### 🎯 Summary Dashboard
- Integrates all three modules into one actionable view per product
- Side-by-side comparison of all three products (Milk, Bread, Orange Juice)
- This week's recommended action: price adjustment, demand forecast, and order quantity

---

## Tech Stack

| Module | Technology |
|---|---|
| Demand Forecasting | `statsmodels` Holt-Winters ExponentialSmoothing |
| Dynamic Pricing | `scipy.optimize` minimize_scalar |
| Inventory Optimization | `ortools` CP-SAT solver |
| Visualization | `plotly` |
| UI | `streamlit` |
| Data | Synthetic data generator (NumPy) |

---

## Project Structure

```
retail-decision-engine/
├── app.py              # Streamlit UI — tabs, sidebar, dashboard
├── data_generator.py   # Synthetic weekly/daily sales data
├── forecasting.py      # Holt-Winters model + pattern decomposition
├── pricing.py          # Price elasticity + SciPy optimization
├── inventory.py        # OR-Tools CP-SAT inventory planner
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/mona2083/retail-decision-engine.git
cd retail-decision-engine
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Demo Data
1. Click **"🎲 Generate & Run Models"** in the sidebar to generate synthetic data and run all models
2. Change the **Random Seed** to generate a different dataset
3. Select a product tab (🥛 Milk / 🍞 Bread / 🍊 Orange Juice)

### Your Own Data
Upload a CSV with the following format:

```csv
date,product_id,sales
2024-01-01,milk,840
2024-01-01,bread,630
2024-01-01,oj,490
2024-01-08,milk,810
```

| Column | Type | Description |
|---|---|---|
| `date` | date | Week start date (YYYY-MM-DD) |
| `product_id` | string | `milk`, `bread`, or `oj` |
| `sales` | integer | Weekly sales quantity |

### Inventory Optimization
1. Go to the **📦 Inventory Optimization** section within any product tab
2. Set current stock, min/max levels, order cost, and holding cost
3. Click **"🚀 Optimize Order Plan"** to run the solver
4. Results appear in the **🎯 Summary Dashboard** at the bottom of the page

---

## How the Models Connect

```
Historical Sales
      │
      ▼
Holt-Winters Forecast (8 weeks)
      │                    │
      ▼                    ▼
OR-Tools Inventory    Price Elasticity
Optimizer             + SciPy Optimizer
      │                    │
      └──────────┬──────────┘
                 ▼
        Summary Dashboard
   (Price · Demand · Order per product)
```

---

## Products & Parameters

| Product | Base Price | Elasticity | Peak Season |
|---|---|---|---|
| 🥛 Milk | $3.99 | -1.8 | Winter |
| 🍞 Bread | $2.49 | -1.4 | November |
| 🍊 Orange Juice | $4.99 | -2.2 | December–January |

---

## Language Support

Toggle between **English** and **日本語** using the language selector in the sidebar.

---

## Author

**Manami Oyama** — AI Engineer / Product Manager  
🌺 Honolulu, Hawaii  
🔗 [Portfolio](https://mona2083.github.io/portfolio-2026/index.html) | [GitHub](https://github.com/mona2083) | [LinkedIn](https://www.linkedin.com/in/manami-oyama/)
