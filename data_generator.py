"""
Retail & Food Service Time-Series Dataset Generator.

Simulates multivariate transactional demand datasets for perishable and retail goods,
incorporating seasonal trends, weekly demand spikes, weather patterns,
inflation curves, holiday calendars, and dynamic promotional discounting.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Core product catalog with physical and economic constraints
PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "PROD_001",
        "name": "Günlük Organik Süt (1L)",
        "category": "Süt & Şarküteri",
        "shelf_life_days": 4,
        "lead_time_days": 1,
        "unit_cost": 28.0,
        "selling_price": 42.0,
        "current_stock": 190,
        "base_demand": 120,
        "weekend_boost": 1.35,
        "weather_sensitivity": "cold_rainy",
        "spoilage_priority": "CRITICAL",
    },
    {
        "id": "PROD_002",
        "name": "Taze Norveç Somon Fileto (500g)",
        "category": "Taze Balık & Restoran",
        "shelf_life_days": 3,
        "lead_time_days": 2,
        "unit_cost": 240.0,
        "selling_price": 390.0,
        "current_stock": 42,
        "base_demand": 45,
        "weekend_boost": 1.75,
        "weather_sensitivity": "weekend_dining",
        "spoilage_priority": "CRITICAL",
    },
    {
        "id": "PROD_003",
        "name": "Hass Avokado (Adet)",
        "category": "Taze Manav",
        "shelf_life_days": 6,
        "lead_time_days": 2,
        "unit_cost": 25.0,
        "selling_price": 45.0,
        "current_stock": 210,
        "base_demand": 85,
        "weekend_boost": 1.30,
        "weather_sensitivity": "sunny",
        "spoilage_priority": "HIGH",
    },
    {
        "id": "PROD_004",
        "name": "Gurme Burger Ekmeği (4'lü)",
        "category": "Unlu Mamüller",
        "shelf_life_days": 5,
        "lead_time_days": 1,
        "unit_cost": 22.0,
        "selling_price": 38.0,
        "current_stock": 280,
        "base_demand": 140,
        "weekend_boost": 1.65,
        "weather_sensitivity": "sunny_outdoor",
        "spoilage_priority": "HIGH",
    },
    {
        "id": "PROD_005",
        "name": "Soğuk Sıkım Zeytinyağı (1L)",
        "category": "Kuru Gıda & Kiler",
        "shelf_life_days": 365,
        "lead_time_days": 5,
        "unit_cost": 280.0,
        "selling_price": 440.0,
        "current_stock": 65,
        "base_demand": 32,
        "weekend_boost": 1.15,
        "weather_sensitivity": "neutral",
        "spoilage_priority": "LOW",
    },
    {
        "id": "PROD_006",
        "name": "Single Origin Çekirdek Kahve (500g)",
        "category": "İçecek & E-Ticaret",
        "shelf_life_days": 180,
        "lead_time_days": 4,
        "unit_cost": 180.0,
        "selling_price": 310.0,
        "current_stock": 95,
        "base_demand": 60,
        "weekend_boost": 1.25,
        "weather_sensitivity": "rainy_cold",
        "spoilage_priority": "LOW",
    },
]


def generate_retail_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2026-10-12",
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Generate synthetic historical daily sales and external exogenous features.

    Args:
        start_date: Timeline start date (YYYY-MM-DD).
        end_date: Timeline end date (YYYY-MM-DD).
        output_dir: Target directory for generated CSV and JSON files.

    Returns:
        pd.DataFrame containing the generated time-series dataset.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1
    dates = [start + timedelta(days=i) for i in range(total_days)]

    np.random.seed(42)
    rows: List[Dict[str, Any]] = []

    # Representative national holiday calendar
    annual_holidays = {
        (1, 1),
        (4, 23),
        (5, 1),
        (5, 19),
        (7, 15),
        (8, 30),
        (10, 29),
    }

    # Extended festive holiday windows
    extended_festivals = {
        "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12",
        "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19",
        "2025-03-30", "2025-03-31", "2025-04-01", "2025-04-02",
        "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09",
        "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23",
        "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30",
    }

    # Weather categorical mapping
    weather_code_map = {
        "Güneşli": 0,
        "Açık": 0,
        "Parçalı Bulutlu": 1,
        "Bulutlu": 2,
        "Yağmurlu": 3,
        "Karlı": 4,
        "Fırtınalı": 5,
    }

    # Generate meteorological & macroeconomic environment
    env_data: Dict[str, Dict[str, Any]] = {}
    for d in dates:
        day_of_year = d.timetuple().tm_yday
        temp_base = 18.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
        temp = float(np.round(temp_base + np.random.normal(0, 3.2), 1))

        if temp < 2:
            weather = np.random.choice(["Karlı", "Bulutlu", "Fırtınalı"], p=[0.6, 0.3, 0.1])
        elif temp > 22:
            weather = np.random.choice(["Güneşli", "Açık", "Parçalı Bulutlu"], p=[0.7, 0.2, 0.1])
        else:
            weather = np.random.choice(["Güneşli", "Bulutlu", "Yağmurlu"], p=[0.4, 0.4, 0.2])

        date_str = d.strftime("%Y-%m-%d")
        is_holiday = 1 if ((d.month, d.day) in annual_holidays or date_str in extended_festivals) else 0

        progress = (d - start).days / total_days
        inflation_idx = float(np.round(1.0 + 0.65 * progress + 0.05 * np.sin(progress * 10), 3))

        env_data[date_str] = {
            "temperature": temp,
            "weather": str(weather),
            "is_holiday": is_holiday,
            "inflation_idx": inflation_idx,
        }

    for prod in PRODUCTS:
        p_id = prod["id"]
        base_d = prod["base_demand"]
        weekend_mult = prod["weekend_boost"]

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            env = env_data[date_str]
            day_of_week = d.weekday()
            is_weekend = 1 if day_of_week in [5, 6] else 0

            # Promotional discount chance
            is_promo = 1 if (np.random.rand() < 0.12 and not env["is_holiday"]) else 0
            discount_pct = float(np.random.choice([0.10, 0.15, 0.20, 0.25])) if is_promo else 0.0

            mult = 1.0

            # Day of week seasonality factors
            if day_of_week in [5, 6]:
                mult *= weekend_mult
            elif day_of_week == 4:
                mult *= 1.18
            elif day_of_week == 1:
                mult *= 0.96
            elif day_of_week == 0:
                mult *= 0.92

            # Meteorological elasticity
            w = env["weather"]
            t = env["temperature"]
            sensitivity = prod["weather_sensitivity"]

            if sensitivity == "cold_rainy":
                if w in ["Yağmurlu", "Karlı", "Fırtınalı"]:
                    mult *= 1.12
            elif sensitivity == "sunny":
                if "Güneşli" in w or "Açık" in w:
                    mult *= 1.20
                elif w in ["Yağmurlu", "Karlı"]:
                    mult *= 0.85
            elif sensitivity == "sunny_outdoor":
                if t > 20 and ("Güneşli" in w or "Açık" in w):
                    mult *= 1.30
                elif w in ["Yağmurlu", "Karlı"]:
                    mult *= 0.75
            elif sensitivity == "weekend_dining":
                if is_weekend and t > 15:
                    mult *= 1.15
            elif sensitivity == "rainy_cold":
                if w in ["Yağmurlu", "Karlı"] or t < 12:
                    mult *= 1.25

            # Holiday demand surge
            if env["is_holiday"]:
                if prod["category"] in ["Süt & Şarküteri", "Unlu Mamüller", "Taze Balık & Restoran"]:
                    mult *= 1.45
                else:
                    mult *= 1.20

            # Price elasticity
            if discount_pct > 0:
                mult *= (1.0 + discount_pct * 1.8)

            # Trend & annual cycles
            time_trend = 1.0 + 0.15 * ((d - start).days / total_days)
            mult *= time_trend

            month = d.month
            season_factor = 1.0 + 0.10 * np.cos(2 * np.pi * (month - 7) / 12)
            mult *= season_factor

            # Stochastic variance
            expected_demand = base_d * mult
            noise = np.random.normal(0, np.sqrt(expected_demand) * 0.9)
            sales = max(5, int(np.round(expected_demand + noise)))

            weather_code = weather_code_map.get(w, 1)

            rows.append({
                "date": date_str,
                "product_id": p_id,
                "product_name": prod["name"],
                "category": prod["category"],
                "sales": sales,
                "temperature": env["temperature"],
                "weather": env["weather"],
                "weather_code": weather_code,
                "is_holiday": env["is_holiday"],
                "is_weekend": is_weekend,
                "day_of_week": day_of_week,
                "month": month,
                "day_of_month": d.day,
                "inflation_idx": env["inflation_idx"],
                "discount_pct": discount_pct,
                "shelf_life_days": prod["shelf_life_days"],
                "lead_time_days": prod["lead_time_days"],
                "unit_cost": prod["unit_cost"],
                "selling_price": prod["selling_price"],
            })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "retail_sales_data.csv")
    meta_path = os.path.join(output_dir, "products_meta.json")

    df.to_csv(csv_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(PRODUCTS, f, ensure_ascii=False, indent=2)

    print(f"[Dataset] Generated {len(df)} records across {len(PRODUCTS)} products.")
    print(f"[Dataset] Saved to {csv_path} and {meta_path}.")
    return df


if __name__ == "__main__":
    generate_retail_dataset()
