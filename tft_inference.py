from __future__ import annotations

import os
# PyTorch/Lightning が MPS を使おうとするのをブロック
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import inspect
import logging
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
        torch.backends.mps.is_built = lambda: False      # type: ignore[method-assign]
    except Exception:
        pass

_disable_mps_for_cpu_inference()

from pytorch_forecasting import TemporalFusionTransformer

# ── PyTorch 2.6+ セキュリティアップデート対策 ──
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        from pytorch_forecasting.data.encoders import (
            EncoderNormalizer, GroupNormalizer, NaNLabelEncoder, MultiNormalizer
        )
        from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE, MAPE
        torch.serialization.add_safe_globals([
            EncoderNormalizer, GroupNormalizer, NaNLabelEncoder, MultiNormalizer,
            QuantileLoss, MAE, RMSE, MAPE
        ])
    except Exception:
        pass

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

def _deep_clean_mps(obj, visited=None):
    """
    テンソル、デバイスオブジェクト、文字列の隅々まで探索し、
    MPS（Mac GPU）の指定を強制的にCPUに置換する完全なクリーナー。
    """
    if visited is None:
        visited = {}
    
    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]

    # 1. テンソル
    if isinstance(obj, torch.Tensor):
        new_obj = obj.cpu()
        visited[obj_id] = new_obj
        return new_obj
        
    # 2. デバイスオブジェクト
    if isinstance(obj, torch.device):
        if obj.type == 'mps':
            new_obj = torch.device('cpu')
            visited[obj_id] = new_obj
            return new_obj
        visited[obj_id] = obj
        return obj
        
    # 3. 文字列（"mps" や "mps:0" などを "cpu" に）
    if isinstance(obj, str):
        if obj in ['mps', 'mps:0']:
            return 'cpu'
        visited[obj_id] = obj
        return obj

    # 4. 辞書
    if isinstance(obj, dict):
        new_dict = {}
        visited[obj_id] = new_dict
        for k, v in obj.items():
            new_dict[_deep_clean_mps(k, visited)] = _deep_clean_mps(v, visited)
        return new_dict
        
    # 5. リスト
    if isinstance(obj, list):
        new_list = []
        visited[obj_id] = new_list
        for v in obj:
            new_list.append(_deep_clean_mps(v, visited))
        return new_list
        
    # 6. タプル
    if isinstance(obj, tuple):
        new_tuple = tuple(_deep_clean_mps(v, visited) for v in obj)
        visited[obj_id] = new_tuple
        return new_tuple

    # 7. カスタムクラスの内部変数 (__dict__)
    if hasattr(obj, "__dict__"):
        visited[obj_id] = obj # 先に登録して無限ループ回避
        for k, v in list(vars(obj).items()):
            try:
                setattr(obj, k, _deep_clean_mps(v, visited))
            except Exception:
                pass
        return obj

    visited[obj_id] = obj
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
        # 1. PyTorch 2.6対策 (weights_only=False) とラムダによる安全なロード
        ckpt = torch.load(
            path, 
            map_location=lambda storage, loc: storage.cpu(),
            weights_only=False
        )
        
        # 2. パラメータの抽出と究極クリーニング
        hparams = ckpt.get("hyper_parameters", {})
        hparams = _deep_clean_mps(hparams)
        dataset_parameters = ckpt.get("dataset_parameters") or hparams.get("dataset_parameters")

        # 版差分で衝突する既知キーを除去
        hparams.pop("monotone_constraints", None)

        # 現在のTFT実装（基底クラス含む）が受け取れる引数だけ残す
        # output_transformer など基底クラスのパラメータも拾うため MRO 全体を走査する
        allowed: set[str] = set()
        for cls in TemporalFusionTransformer.__mro__:
            if cls is object:
                continue
            init = vars(cls).get("__init__")
            if init is None:
                continue
            try:
                allowed.update(inspect.signature(init).parameters)
            except (ValueError, TypeError):
                pass
        allowed.discard("self")
        hparams = {k: v for k, v in hparams.items() if k in allowed}
        
        # 3. PyTorch Lightningのバグを【完全にバイパス】して直接インスタンス化
        model = TemporalFusionTransformer(**hparams)

        # 3.5 predict() に必要な dataset_parameters を復元
        if dataset_parameters is not None:
            dataset_parameters = _deep_clean_mps(dataset_parameters)
            model.dataset_parameters = dataset_parameters
        else:
            msg = "TFT checkpoint missing dataset_parameters; fallback mode enabled"
            logger.warning(msg)
            return None, "error", msg
        
        # 4. 重み（State Dict）のクリーニングとロード
        state_dict = ckpt.get("state_dict", {})
        state_dict = _deep_clean_mps(state_dict)
        model.load_state_dict(state_dict, strict=False)
        
        # 5. 強制的にCPUに固定し、推論モードに
        model = model.cpu()
        model.eval()

        return model, "ok", ""
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
    if getattr(model, "dataset_parameters", None) is None:
        raise ValueError("TFT model has no dataset_parameters for predict()")

    
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