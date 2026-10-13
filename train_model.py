"""
Deep Learning Multivariate Time-Series Forecasting Model Training.

Trains a hybrid Bidirectional LSTM with Multi-Head Self-Attention and GRU architecture
on historical retail sales and exogenous environmental features.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEQUENCE_LENGTH: int = 14
FORECAST_HORIZON: int = 7


def prepare_data(
    csv_path: str = "data/retail_sales_data.csv",
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, List[str], pd.DataFrame]:
    """
    Extract, normalize, and construct sliding sequence windows for time-series forecasting.

    Args:
        csv_path: Path to the transactional CSV dataset.

    Returns:
        Tuple containing input sequences (X), target sequences (y), target scaler,
        feature scaler, feature names, and raw dataframe.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)

    # Cyclical harmonic encodings for periodic calendar signals
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # One-hot product entity identifiers
    product_dummies = pd.get_dummies(df["product_id"], prefix="prod", dtype=float)
    df = pd.concat([df, product_dummies], axis=1)

    feature_cols = [
        "sales",
        "temperature",
        "weather_code",
        "is_holiday",
        "is_weekend",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "discount_pct",
        "inflation_idx",
    ] + [c for c in product_dummies.columns]

    sales_scaler = MinMaxScaler(feature_range=(0, 1))
    df["sales_scaled"] = sales_scaler.fit_transform(df[["sales"]])

    other_features = [c for c in feature_cols if c != "sales"]
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_feats = feat_scaler.fit_transform(df[other_features])
    scaled_feat_df = pd.DataFrame(scaled_feats, columns=other_features)

    df_scaled = pd.concat(
        [df[["date", "product_id", "product_name"]], df[["sales_scaled"]], scaled_feat_df],
        axis=1,
    )

    all_feature_names = ["sales_scaled"] + other_features

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    product_ids = df["product_id"].unique()

    for pid in product_ids:
        pdf = df_scaled[df_scaled["product_id"] == pid].reset_index(drop=True)
        feat_array = pdf[all_feature_names].values
        sales_array = pdf["sales_scaled"].values

        n_samples = len(pdf) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1
        for i in range(n_samples):
            x_window = feat_array[i : i + SEQUENCE_LENGTH]
            y_window = sales_array[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + FORECAST_HORIZON]
            x_list.append(x_window)
            y_list.append(y_window)

    x_arr = np.array(x_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)

    return x_arr, y_arr, sales_scaler, feat_scaler, all_feature_names, df


def build_temporal_attention_model(
    input_shape: Tuple[int, int],
    forecast_horizon: int = 7,
) -> tf.keras.Model:
    """
    Construct a hybrid Bidirectional LSTM + Multi-Head Self-Attention neural network.

    Args:
        input_shape: Shape tuple (sequence_length, feature_dimension).
        forecast_horizon: Output forecast horizon in days.

    Returns:
        Compiled Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="time_series_input")

    # Bidirectional LSTM captures past and future temporal context within lookback
    bilstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.0)
    )(inputs)
    bilstm = layers.LayerNormalization()(bilstm)

    # Multi-head attention captures inter-day relationships (e.g. periodic weekly spikes)
    attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(bilstm, bilstm)
    attention_out = layers.Dropout(0.1)(attention_out)
    residual = layers.Add()([bilstm, attention_out])
    residual = layers.LayerNormalization()(residual)

    # GRU layer encodes high-level temporal dynamics
    gru = layers.GRU(32, return_sequences=False, dropout=0.1)(residual)

    # Dense projection head
    dense1 = layers.Dense(64, activation="relu")(gru)
    dense1 = layers.Dropout(0.15)(dense1)
    dense2 = layers.Dense(32, activation="relu")(dense1)

    outputs = layers.Dense(forecast_horizon, activation="relu", name="forecast_output")(dense2)

    model = models.Model(inputs=inputs, outputs=outputs, name="RetailDemandForecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae", "mse"],
    )
    return model


def train_and_evaluate(
    epochs: int = 30,
    batch_size: int = 32,
    model_dir: str = "saved_model",
) -> Dict[str, Any]:
    """
    Train and evaluate the demand forecaster model, saving artifacts to disk.

    Args:
        epochs: Maximum training epochs.
        batch_size: Batch size for mini-batch gradient descent.
        model_dir: Directory where weights and scalers will be saved.

    Returns:
        Dictionary containing validation evaluation metrics and training history.
    """
    print("[Pipeline] Loading and preprocessing dataset...")
    x_data, y_data, sales_scaler, feat_scaler, feature_names, _ = prepare_data()
    print(f"[Pipeline] Feature tensor shape: {x_data.shape}, Target tensor shape: {y_data.shape}")

    # Hold-out validation split
    split_idx = int(len(x_data) * 0.85)
    train_indices = np.random.RandomState(42).permutation(split_idx)
    x_train, y_train = x_data[train_indices], y_data[train_indices]
    x_val, y_val = x_data[split_idx:], y_data[split_idx:]

    print(f"[Pipeline] Train samples: {len(x_train)} | Validation samples: {len(x_val)}")

    input_shape = (SEQUENCE_LENGTH, len(feature_names))
    model = build_temporal_attention_model(input_shape, FORECAST_HORIZON)

    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_demand_model.keras")

    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    print("[Pipeline] Training TensorFlow model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=1,
    )

    # Compute validation performance metrics
    val_preds_scaled = model.predict(x_val)
    val_preds_orig = sales_scaler.inverse_transform(val_preds_scaled)
    y_val_orig = sales_scaler.inverse_transform(y_val)

    mae = float(np.mean(np.abs(val_preds_orig - y_val_orig)))
    rmse = float(np.sqrt(np.mean((val_preds_orig - y_val_orig) ** 2)))

    non_zero = y_val_orig > 0
    mape = float(np.mean(np.abs((val_preds_orig[non_zero] - y_val_orig[non_zero]) / y_val_orig[non_zero])) * 100.0)

    ss_res = np.sum((y_val_orig - val_preds_orig) ** 2)
    ss_tot = np.sum((y_val_orig - np.mean(y_val_orig)) ** 2)
    r2_score = float(1.0 - (ss_res / ss_tot))

    print("\n" + "=" * 50)
    print("MODEL EVALUATION SUMMARY:")
    print(f"MAE  (Mean Absolute Error):        {mae:.2f} units")
    print(f"RMSE (Root Mean Squared Error):    {rmse:.2f} units")
    print(f"MAPE (Mean Abs Percentage Error):  {mape:.2f}%")
    print(f"Accuracy Metric (100 - MAPE):      {max(0.0, 100.0 - mape):.2f}%")
    print(f"R² Score:                          {r2_score:.4f}")
    print("=" * 50 + "\n")

    # Save model and normalization transforms
    model.save(os.path.join(model_dir, "demand_forecast_model.keras"))
    joblib.dump(sales_scaler, os.path.join(model_dir, "sales_scaler.joblib"))
    joblib.dump(feat_scaler, os.path.join(model_dir, "feat_scaler.joblib"))

    train_history = {
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "mae": [float(v) for v in history.history["mae"]],
        "val_mae": [float(v) for v in history.history["val_mae"]],
        "epochs_trained": len(history.history["loss"]),
        "metrics": {
            "mae_units": round(mae, 2),
            "rmse_units": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "accuracy_pct": round(max(0.0, 100.0 - mape), 2),
            "r2_score": round(r2_score, 4),
        },
        "feature_names": feature_names,
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
    }

    with open(os.path.join(model_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)

    print(f"[Pipeline] Artifacts successfully serialized to '{model_dir}/'.")
    return train_history


if __name__ == "__main__":
    train_and_evaluate(epochs=30)
