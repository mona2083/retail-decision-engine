import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_forecast(weekly_sales: pd.Series, forecast_weeks: int = 8) -> dict:
    model = ExponentialSmoothing(
        weekly_sales,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated",
    )
    fit        = model.fit(optimized=True)
    forecast   = fit.forecast(forecast_weeks)
    fitted     = fit.fittedvalues

    resid       = weekly_sales - fitted
    resid_std   = resid.std()
    lower       = forecast - 1.96 * resid_std
    upper       = forecast + 1.96 * resid_std

    return {
        "history":       weekly_sales,
        "fitted":        fitted,
        "forecast":      forecast,
        "lower":         lower.clip(lower=0),
        "upper":         upper,
        "resid_std":     resid_std,
        "mape":          (abs(resid) / weekly_sales.replace(0, np.nan)).mean() * 100,
        "fit":           fit,
    }


def decompose_weekly_patterns(daily_sales: pd.DataFrame, product_id: str) -> dict:
    df = daily_sales[daily_sales["product_id"] == product_id].copy()

    weekday_avg = df.groupby("weekday")["sales"].mean()
    order       = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = weekday_avg.reindex([d for d in order if d in weekday_avg.index])

    df["month"]   = df["date"].dt.month
    monthly_avg   = df.groupby("month")["sales"].mean()

    return {
        "weekday_avg": weekday_avg,
        "monthly_avg": monthly_avg,
    }
