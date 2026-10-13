"""
Inventory Optimization and Decision Intelligence Engine.

Integrates deep learning time-series forecasts with inventory control algorithms,
computing safety stocks, reorder points, stockout exposure, and perishable waste risks.
"""

from datetime import datetime, timedelta
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

TURKISH_DAY_NAMES: List[str] = [
    "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"
]
WEATHER_NAMES: Dict[int, str] = {
    0: "Güneşli / Açık",
    1: "Parçalı Bulutlu",
    2: "Bulutlu",
    3: "Yağmurlu",
    4: "Karlı",
    5: "Fırtınalı",
}


class InventoryForecastingEngine:
    """
    Core analytics engine for demand forecasting and inventory replenishment.
    """

    def __init__(self, model_dir: str = "saved_model", data_dir: str = "data") -> None:
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model: Optional[tf.keras.Model] = None
        self.sales_scaler: Optional[Any] = None
        self.feat_scaler: Optional[Any] = None
        self.metrics: Dict[str, Any] = {}
        self.products: List[Dict[str, Any]] = []
        self.df: Optional[pd.DataFrame] = None
        self.load_assets()

    def load_assets(self) -> None:
        """Load trained neural network, scalers, and metadata."""
        model_path = os.path.join(self.model_dir, "demand_forecast_model.keras")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)

        sales_scaler_path = os.path.join(self.model_dir, "sales_scaler.joblib")
        if os.path.exists(sales_scaler_path):
            self.sales_scaler = joblib.load(sales_scaler_path)

        feat_scaler_path = os.path.join(self.model_dir, "feat_scaler.joblib")
        if os.path.exists(feat_scaler_path):
            self.feat_scaler = joblib.load(feat_scaler_path)

        metrics_path = os.path.join(self.model_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        meta_path = os.path.join(self.data_dir, "products_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.products = json.load(f)

        data_path = os.path.join(self.data_dir, "retail_sales_data.csv")
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            self.df["date"] = pd.to_datetime(self.df["date"])

    def get_products(self) -> List[Dict[str, Any]]:
        """Return the list of configured products."""
        return self.products

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific product metadata dict by its ID."""
        for p in self.products:
            if p["id"] == product_id:
                return p
        return None

    def _prepare_recent_sequence(
        self, product_id: str
    ) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
        """Extract the most recent 14-day window for inference."""
        if self.df is None:
            raise RuntimeError("Underlying dataset is not loaded.")

        pdf = self.df[self.df["product_id"] == product_id].sort_values(by="date").reset_index(drop=True)
        recent_14 = pdf.tail(14).copy()

        recent_14["day_sin"] = np.sin(2 * np.pi * recent_14["day_of_week"] / 7.0)
        recent_14["day_cos"] = np.cos(2 * np.pi * recent_14["day_of_week"] / 7.0)
        recent_14["month_sin"] = np.sin(2 * np.pi * recent_14["month"] / 12.0)
        recent_14["month_cos"] = np.cos(2 * np.pi * recent_14["month"] / 12.0)

        all_pids = [p["id"] for p in self.products]
        for pid in all_pids:
            recent_14[f"prod_{pid}"] = 1.0 if pid == product_id else 0.0

        feature_cols = self.metrics.get("feature_names", [])
        if not feature_cols:
            return None, recent_14

        recent_14["sales_scaled"] = self.sales_scaler.transform(recent_14[["sales"]])
        other_features = [c for c in feature_cols if c != "sales_scaled"]
        recent_14[other_features] = self.feat_scaler.transform(recent_14[other_features])

        x_seq = recent_14[feature_cols].values.reshape(1, 14, len(feature_cols)).astype(np.float32)
        return x_seq, pdf

    def predict_product_forecast(
        self, product_id: str, scenario: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate 7-day demand predictions and inventory risk evaluation.

        Args:
            product_id: Target product identifier.
            scenario: Optional exogenous variables override for what-if simulation.

        Returns:
            Structured forecast and risk assessment payload.
        """
        prod = self.get_product_by_id(product_id)
        if not prod:
            raise ValueError(f"Product '{product_id}' not found.")

        x_seq, pdf = self._prepare_recent_sequence(product_id)
        if x_seq is None or self.model is None or self.sales_scaler is None:
            raise RuntimeError("Model or feature transformers are unavailable.")

        raw_pred_scaled = self.model.predict(x_seq, verbose=0)
        pred_units = self.sales_scaler.inverse_transform(raw_pred_scaled)[0]
        pred_units = np.maximum(5.0, pred_units)

        # Apply what-if elasticity multipliers if specified
        if scenario:
            target_weather = scenario.get("weather_code", 1)
            target_temp = scenario.get("temperature", 20.0)
            target_discount = scenario.get("discount_pct", 0.0)
            target_holiday = scenario.get("is_holiday", 0)

            mult = 1.0
            if target_discount > 0:
                mult *= 1.0 + target_discount * 1.8
            if target_holiday:
                mult *= 1.35

            sensitivity = prod.get("weather_sensitivity", "neutral")
            if sensitivity == "cold_rainy" and (target_weather in [3, 4, 5] or target_temp < 12):
                mult *= 1.15
            elif sensitivity == "sunny" and target_weather in [0, 1] and target_temp > 20:
                mult *= 1.20
            elif sensitivity == "sunny_outdoor" and target_temp > 22 and target_weather in [0, 1]:
                mult *= 1.30
            elif sensitivity == "weekend_dining":
                mult *= 1.12
            elif sensitivity == "rainy_cold" and target_weather in [3, 4, 5]:
                mult *= 1.25

            pred_units = pred_units * mult

        last_date = pdf["date"].max()
        future_days: List[Dict[str, Any]] = []
        next_tuesday_info: Optional[Dict[str, Any]] = None

        mae_err = self.metrics.get("metrics", {}).get("mae_units", 11.0)

        for i in range(7):
            cur_date = last_date + timedelta(days=i + 1)
            day_idx = cur_date.weekday()
            day_name = TURKISH_DAY_NAMES[day_idx]
            val = round(float(pred_units[i]), 1)

            lower_bound = max(0, round(val - 1.28 * mae_err, 1))
            upper_bound = round(val + 1.28 * mae_err, 1)

            day_item = {
                "day_index": i + 1,
                "date": cur_date.strftime("%Y-%m-%d"),
                "day_name": day_name,
                "is_tuesday": (day_idx == 1),
                "is_weekend": (day_idx in [5, 6]),
                "predicted_sales": int(round(val)),
                "lower_bound": int(round(lower_bound)),
                "upper_bound": int(round(upper_bound)),
            }
            future_days.append(day_item)

            if day_idx == 1 and next_tuesday_info is None:
                next_tuesday_info = day_item

        if not next_tuesday_info:
            next_tuesday_info = future_days[1]

        # Inventory Replenishment & Buffer Calculations
        total_7day_demand = sum(d["predicted_sales"] for d in future_days)
        daily_avg_demand = total_7day_demand / 7.0

        recent_std = float(pdf.tail(30)["sales"].std())
        lead_time = prod["lead_time_days"]
        shelf_life = prod["shelf_life_days"]
        current_stock = prod["current_stock"]
        unit_cost = prod["unit_cost"]
        selling_price = prod["selling_price"]

        # Safety Stock formula: Z=1.65 (95% service level) * sigma_d * sqrt(L)
        safety_stock = int(round(1.65 * recent_std * np.sqrt(lead_time)))
        # Reorder Point formula: (avg_demand * lead_time) + safety_stock
        reorder_point = int(round((daily_avg_demand * lead_time) + safety_stock))

        # Stockout Exposure Analysis
        stockout_risk = False
        depletion_day_name = None
        depletion_date = None
        stock_remaining = current_stock
        lost_sales = 0

        for d in future_days:
            if stock_remaining >= d["predicted_sales"]:
                stock_remaining -= d["predicted_sales"]
            else:
                if not stockout_risk:
                    stockout_risk = True
                    depletion_day_name = d["day_name"]
                    depletion_date = d["date"]
                shortage = d["predicted_sales"] - stock_remaining
                lost_sales += shortage
                stock_remaining = 0

        lost_revenue_try = round(lost_sales * selling_price, 2)
        recommended_reorder_qty = max(0, int(round((total_7day_demand + safety_stock) - current_stock)))

        # Perishable Spoilage Risk Analysis
        waste_risk = False
        spoiled_units = 0
        waste_loss_try = 0.0

        if shelf_life <= 7:
            demand_within_shelf = sum(
                future_days[i]["predicted_sales"] for i in range(min(shelf_life, len(future_days)))
            )
            if current_stock > demand_within_shelf:
                waste_risk = True
                spoiled_units = int(current_stock - demand_within_shelf)
                waste_loss_try = round(spoiled_units * unit_cost, 2)

        # Operational Action Guidance
        action_type = "OPTIMAL"
        action_text = "Stok seviyesi ve tahmin edilen talep dengeli. Rutin takip önerilir."

        if stockout_risk:
            action_type = "STOCKOUT_ALERT"
            action_text = (
                f"DİKKAT: Stok {depletion_day_name} ({depletion_date}) günü tükenecek! "
                f"{lost_revenue_try:,.0f} ₺ ciro kaybını önlemek için derhal en az "
                f"{recommended_reorder_qty} adet tedarik siparişi verin."
            )
        elif waste_risk:
            action_type = "SPOILAGE_ALERT"
            action_text = (
                f"ZAYİAT ALARMI: Raf ömrü {shelf_life} gün olan {prod['name']} için "
                f"{spoiled_units} adet ürünün bozulma riski var ({waste_loss_try:,.0f} ₺ maliyet). "
                f"Acil %15-%20 Flaş İndirim tanımlayarak stoğu eritin!"
            )
        elif current_stock <= reorder_point:
            action_type = "REORDER_WARNING"
            action_text = (
                f"Sipariş Noktası Eşiği: Stok ({current_stock} adet), kritik ROP seviyesine "
                f"({reorder_point} adet) yaklaştı. Tedarik süresi ({lead_time} gün) "
                f"göz önüne alınarak yeni sipariş oluşturulmalı."
            )

        recent_30 = pdf.tail(30)[["date", "sales", "temperature", "weather"]].to_dict(orient="records")
        for r in recent_30:
            r["date"] = r["date"].strftime("%Y-%m-%d")

        return {
            "product": prod,
            "next_tuesday_forecast": {
                "date": next_tuesday_info["date"],
                "day_name": next_tuesday_info["day_name"],
                "predicted_sales": next_tuesday_info["predicted_sales"],
                "lower_bound": next_tuesday_info["lower_bound"],
                "upper_bound": next_tuesday_info["upper_bound"],
                "message": (
                    f"Gelecek Salı ({next_tuesday_info['date']}): "
                    f"{next_tuesday_info['predicted_sales']} adet satış beklenmektedir."
                ),
            },
            "future_7days": future_days,
            "recent_30days": recent_30,
            "inventory_analytics": {
                "current_stock": current_stock,
                "total_7day_demand": total_7day_demand,
                "daily_avg_demand": round(daily_avg_demand, 1),
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "stockout_risk": stockout_risk,
                "depletion_day": depletion_day_name,
                "depletion_date": depletion_date,
                "lost_sales_units": lost_sales,
                "lost_revenue_try": lost_revenue_try,
                "recommended_reorder_qty": recommended_reorder_qty,
                "waste_risk": waste_risk,
                "spoiled_units": spoiled_units,
                "waste_loss_try": waste_loss_try,
                "action_type": action_type,
                "action_recommendation": action_text,
            },
        }

    def get_executive_overview(self) -> Dict[str, Any]:
        """Aggregate cross-catalog analytics for executive reporting."""
        all_results: List[Dict[str, Any]] = []
        total_inventory_value = 0.0
        total_waste_loss = 0.0
        total_lost_revenue = 0.0
        total_forecast_demand = 0
        stockout_count = 0
        waste_count = 0

        for p in self.products:
            res = self.predict_product_forecast(p["id"])
            all_results.append(res)

            inv = res["inventory_analytics"]
            total_inventory_value += p["current_stock"] * p["selling_price"]
            total_waste_loss += inv["waste_loss_try"]
            total_lost_revenue += inv["lost_revenue_try"]
            total_forecast_demand += inv["total_7day_demand"]

            if inv["stockout_risk"]:
                stockout_count += 1
            if inv["waste_risk"]:
                waste_count += 1

        accuracy = self.metrics.get("metrics", {}).get("accuracy_pct", 86.5)
        mape = self.metrics.get("metrics", {}).get("mape_pct", 13.5)
        mae = self.metrics.get("metrics", {}).get("mae_units", 11.2)

        return {
            "kpis": {
                "total_inventory_value_try": round(total_inventory_value, 2),
                "total_waste_risk_try": round(total_waste_loss, 2),
                "total_lost_revenue_risk_try": round(total_lost_revenue, 2),
                "total_7day_demand_units": total_forecast_demand,
                "stockout_risk_products": stockout_count,
                "waste_risk_products": waste_count,
                "model_accuracy_pct": accuracy,
                "model_mape_pct": mape,
                "model_mae_units": mae,
            },
            "products_summary": [
                {
                    "id": r["product"]["id"],
                    "name": r["product"]["name"],
                    "category": r["product"]["category"],
                    "current_stock": r["product"]["current_stock"],
                    "next_tuesday_sales": r["next_tuesday_forecast"]["predicted_sales"],
                    "total_7day_sales": r["inventory_analytics"]["total_7day_demand"],
                    "reorder_point": r["inventory_analytics"]["reorder_point"],
                    "action_type": r["inventory_analytics"]["action_type"],
                    "action_recommendation": r["inventory_analytics"]["action_recommendation"],
                    "stockout_risk": r["inventory_analytics"]["stockout_risk"],
                    "waste_risk": r["inventory_analytics"]["waste_risk"],
                }
                for r in all_results
            ],
        }


if __name__ == "__main__":
    engine = InventoryForecastingEngine()
    test_res = engine.predict_product_forecast("PROD_001")
    print(f"Product: {test_res['product']['name']}")
    print(f"Tuesday Forecast: {test_res['next_tuesday_forecast']['predicted_sales']} units")
    print(f"Stockout Risk: {test_res['inventory_analytics']['stockout_risk']}")
