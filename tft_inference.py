from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import logging
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import torch

logger = logging.getLogger(__name__)

def _disable_mps_for_cpu_inference() -> None:
    """Apple Silicon で Lightning/torchmetrics が MPS 上にテンソルを作ろうとして落ちるのを防ぐ"""
    if not hasattr(torch.backends, "mps"):
        return
    try:
        torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
    except Exception:
        pass

# Lightning を import する前に MPS を無効扱い
_disable_mps_for_cpu_inference()

from pytorch_forecasting import TemporalFusionTransformer

def _force_default_device_cpu() -> None:
    try:
        torch.set_default_device("cpu")
    except AttributeError:
        pass

_PACKAGE_DIR = Path(__file__).resolve().parent

def resolve_tft_checkpoint_path(explicit: str | None = None) -> Path:
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
    return str(resolve_tft_checkpoint_path())

def _sanitize_mps(obj):
    """
    【重要パッチ】hyper_parameters の奥底（Loss関数など）に潜む
    MPS(Mac GPU)指定のテンソルを見つけ出し、強制的にCPUへ書き換える
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _sanitize_mps(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_mps(v) for v in obj]
    if hasattr(obj, "__dict__"):
        try:
            for k, v in vars(obj).items():
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.cpu())
                elif isinstance(v, (list, dict)):
                    setattr(obj, k, _sanitize_mps(v))
        except Exception:
            pass
    return obj

def load_tft_model(
    model_path: str | None = None,
) -> tuple[TemporalFusionTransformer | None, str, str]:
    path = resolve_tft_checkpoint_path(model_path)
    if not path.is_file():
        logger.info("TFT checkpoint not found: %s", path)
        return None, "missing", str(path)

    _disable_mps_for_cpu_inference()
    _force_default_device_cpu()

    try:
        # 1. まず標準の機能でチェックポイントをメモリ上にロード
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        
        # 2. PyTorch Forecasting特有の問題（hparams内のMPS残留）を強制クリーニング
        if "hyper_parameters" in ckpt:
            ckpt["hyper_parameters"] = _sanitize_mps(ckpt["hyper_parameters"])
        
        # 3. クリーニング済みのデータを「安全な一時チェックポイント」として保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp:
            torch.save(ckpt, tmp.name)
            safe_ckpt_path = tmp.name
        
        # 4. 完全にクリーンになった一時ファイルからモデルをロード
        model = TemporalFusionTransformer.load_from_checkpoint(
            safe_ckpt_path,
            map_location=torch.device("cpu")
        )
        
        # 5. 用済みのゴミ一時ファイルを削除
        try:
            os.remove(safe_ckpt_path)
        except Exception:
            pass

        return model.cpu(), "ok", ""
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