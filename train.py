import os
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

# data_generator.py からデータ生成関数をインポート（同じディレクトリにある前提）
from data_generator import generate_daily_sales_with_price

def prepare_data():
    # データの生成と前処理
    df = generate_daily_sales_with_price(1095, random_state=42)
    
    # TFT必須要件: 連続した時間インデックスの作成
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("product_id").cumcount()
    
    # カテゴリ変数は文字列にキャスト（要件）
    df["product_id"] = df["product_id"].astype(str)
    df["weekday"] = df["weekday"].astype(str)
    df["month"] = df["month"].astype(str)
    df["is_weekend"] = df["is_weekend"].astype(str)
    
    # ターゲット（sales）をfloatに変換（ロス計算のため）
    df["sales"] = df["sales"].astype(float)
    return df

def train_tft():
    # 1. 再現性のための完全なシード固定（実務必須要件）
    pl.seed_everything(42, workers=True)

    df = prepare_data()

    # パラメータ設定
    max_prediction_length = 56  # 予測期間：8週間（56日）
    max_encoder_length = 90     # 入力期間：過去約3ヶ月
    training_cutoff = df["time_idx"].max() - max_prediction_length

    # 2. TimeSeriesDataSetの定義（データリーク防止の要）
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["product_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["product_id"],
        time_varying_known_categoricals=["weekday", "month", "is_weekend"],
        time_varying_known_reals=["price", "time_idx"], # 未来の価格は計画として「既知」
        time_varying_unknown_reals=["sales"],           # 未来の売上は「未知」
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 3. 検証データの作成（絶対に訓練データで評価しないための分割）
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    # 4. TFTモデルの定義
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,          # 過学習防止のため小規模に設定
        attention_head_size=1,
        dropout=0.2,             # 過学習防止
        hidden_continuous_size=8,
        loss=QuantileLoss(),     # 信頼区間（10%, 50%, 90%等）を出力
        optimizer="Adam",
    )

    # 5. コールバック設定（Early Stoppingとモデル保存）
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="models", filename="tft_best_model", monitor="val_loss", mode="min"
    )

    # 6. Trainerの実行
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto", # Mac(Apple Silicon)の場合はMPSが自動選択される可能性があります
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=True, # 再現性の担保
    )

    print("🚀 モデルの学習を開始します...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print(f"✅ 学習完了。最良モデルが保存されました: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train_tft()