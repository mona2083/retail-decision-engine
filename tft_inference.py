from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

logger = logging.getLogger(__name__)

# アプリ本体と同じディレクトリ基準（Streamlit Cloud で cwd が違っても .ckpt を探せる）
_PACKAGE_DIR = Path(__file__).resolve().parent


def resolve_tft_checkpoint_path(explicit: str | None = None) -> Path:
    """チェックポイントの絶対パス。相対パスは `tft_inference.py` があるディレクトリ基準で解決。"""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (_PACKAGE_DIR / p).resolve()
        else:
            p = p.resolve()
        return p
    env = os.environ.get("TFT_MODEL_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (_PACKAGE_DIR / p).resolve()
        else:
            p = p.resolve()
        return p
    return (_PACKAGE_DIR / "models" / "tft_best_model.ckpt").resolve()


def default_tft_checkpoint_path() -> str:
    """UI 表示用。環境変数 TFT_MODEL_PATH または既定の `models/tft_best_model.ckpt`（絶対パス化）。"""
    return str(resolve_tft_checkpoint_path())


def load_tft_model(
    model_path: str | None = None,
) -> tuple[TemporalFusionTransformer | None, str, str]:
    """モデルをロード。

    Returns:
        (model, status, detail)
        - status: ``\"ok\"`` | ``\"missing\"`` | ``\"error\"``
        - detail: 失敗時は絶対パス（missing）またはエラーメッセージ（error）。成功時は空文字。
    """
    path = resolve_tft_checkpoint_path(model_path)
    if not path.is_file():
        logger.info("TFT checkpoint not found: %s", path)
        return None, "missing", str(path)

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(path),
            map_location="cpu",
        )
        return model, "ok", ""
    except NotImplementedError as exc:
        logger.warning("TFT load failed (NotImplementedError): %s", exc)
        return None, "error", f"NotImplementedError: {exc}"
    except Exception as exc:
        logger.warning("TFT load failed: %s", exc)
        return None, "error", f"{type(exc).__name__}: {exc}"


def predict_dynamic_demand(
    model: TemporalFusionTransformer,
    daily_df: pd.DataFrame,
    product_id: str,
    planned_price: float,
    encoder_days: int = 90,
    decoder_days: int = 56,
) -> pd.DataFrame:
    """過去のデータと未来の計画価格を結合し、TFTで推論を行う"""

    prod_df = daily_df[daily_df["product_id"] == product_id].copy()
    prod_df = prod_df.sort_values("date").reset_index(drop=True)
    history_df = prod_df.tail(encoder_days).copy()

    last_date = history_df["date"].max()
    last_time_idx = history_df["time_idx"].max()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=decoder_days, freq="1D"
    )

    future_rows = []
    for i, dt in enumerate(future_dates):
        future_rows.append(
            {
                "date": dt,
                "product_id": product_id,
                "price": planned_price,
                "sales": 0.0,
                "weekday": str(dt.day_name()),
                "month": str(dt.month),
                "is_weekend": str(1 if dt.dayofweek >= 5 else 0),
                "time_idx": last_time_idx + 1 + i,
            }
        )
    future_df = pd.DataFrame(future_rows)

    inference_df = pd.concat([history_df, future_df], ignore_index=True)

    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            inference_df,
            mode="raw",
            return_x=False,
            trainer_kwargs={"accelerator": "cpu"},
        )

    median_pred = predictions["prediction"][0, :, 1].cpu().numpy()

    future_df["forecast_sales"] = np.maximum(median_pred, 0)

    weekly_forecast = (
        future_df.groupby(pd.Grouper(key="date", freq="W-MON"))
        .agg(forecast_sales=("forecast_sales", "sum"))
        .reset_index()
    )

    return weekly_forecast
