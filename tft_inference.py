import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

def load_tft_model(model_path: str = "models/tft_best_model.ckpt") -> TemporalFusionTransformer:
    """モデルをロード。推論時のオーバーヘッドを防ぐため明示的にCPUにマッピング。"""
    return TemporalFusionTransformer.load_from_checkpoint(
        model_path, 
        map_location=torch.device("cpu")
    )

def predict_dynamic_demand(
    model: TemporalFusionTransformer, 
    daily_df: pd.DataFrame, 
    product_id: str, 
    planned_price: float, 
    encoder_days: int = 90, 
    decoder_days: int = 56
) -> pd.DataFrame:
    """過去のデータと未来の計画価格を結合し、TFTで推論を行う"""
    
    # 1. 過去データ（Encoder系列）の抽出
    prod_df = daily_df[daily_df["product_id"] == product_id].copy()
    prod_df = prod_df.sort_values("date").reset_index(drop=True)
    history_df = prod_df.tail(encoder_days).copy()
    
    # 2. 未来データ（Decoder系列＝シナリオ）の構築
    last_date = history_df["date"].max()
    last_time_idx = history_df["time_idx"].max()
    
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=decoder_days, freq="D")
    
    future_rows = []
    for i, dt in enumerate(future_dates):
        future_rows.append({
            "date": dt,
            "product_id": product_id,
            "price": planned_price,  # ユーザーがスライダーで設定した未来の計画価格
            "sales": 0.0,            # 【重要】未来の売上は絶対に未知のため0で埋める
            "weekday": str(dt.day_name()),
            "month": str(dt.month),
            "is_weekend": str(1 if dt.dayofweek >= 5 else 0),
            "time_idx": last_time_idx + 1 + i
        })
    future_df = pd.DataFrame(future_rows)
    
    # 3. 結合と推論の実行
    inference_df = pd.concat([history_df, future_df], ignore_index=True)
    
    # TFTモデルによる推論（モードを検証モードに固定し、非決定的なドロップアウトを無効化）
    model.eval()
    with torch.no_grad():
        predictions = model.predict(inference_df, mode="raw", return_x=False)
    
    # 中央値（50パーセンタイル）を予測値として抽出（※QuantileLossのインデックスは学習設定に依存）
    median_pred = predictions["prediction"][0, :, 1].cpu().numpy()  # [batch, time, quantile]
    
    future_df["forecast_sales"] = np.maximum(median_pred, 0) # 負の売上をクリップ
    
    # UI表示用に週次に集約
    weekly_forecast = future_df.groupby(pd.Grouper(key='date', freq='W-MON')).agg(
        forecast_sales=('forecast_sales', 'sum')
    ).reset_index()
    
    return weekly_forecast
