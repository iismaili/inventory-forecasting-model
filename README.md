# DemandFlow AI: Enterprise Retail & Warehouse Demand Forecasting Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B%20%2F%20Keras%203-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Bi--LSTM%20%2B%20Attention%20%2B%20GRU-indigo.svg)](#neural-network-architecture)

> **Enterprise-grade time-series forecasting and inventory optimization platform built with TensorFlow/Keras and FastAPI.** Designed for supermarkets, restaurant chains, and e-commerce distribution warehouses to eliminate stockout revenue loss and minimize perishable goods spoilage through multi-step multivariate deep learning.

---

## 📌 Executive Summary & Business Impact

Retailers and commercial kitchens face two major profit-eroding threats:
1. **Stockouts & Under-stocking:** Running out of inventory causes immediate loss of sales, cart abandonment, and customer churn.
2. **Over-stocking & Perishable Spoilage:** Over-purchasing goods with short shelf lives (dairy, fresh produce, seafood, bakery) leads to severe inventory write-offs and food waste.

**DemandFlow AI** delivers accurate, multi-step 7-day demand predictions using historical sales patterns, weather fluctuations, holiday calendars, macroeconomic inflation, and promotional elasticity. It translates raw neural network outputs into automated inventory replenishment orders (Reorder Point & Safety Stock) and proactive markdown recommendations.

---

## 🧠 Neural Network Architecture

```
[Lookback Window: 14 Days x N Features]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ Bidirectional LSTM (64 units) │  <-- Captures temporal sequences forwards & backwards
   └──────────────┬───────────────┘
                  │ LayerNormalization
                  ▼
   ┌──────────────────────────────┐
   │ Multi-Head Attention (4 Hds) │  <-- Discovers long-range day-of-week correlations
   └──────────────┬───────────────┘
                  │ Residual Connection & LayerNormalization
                  ▼
   ┌──────────────────────────────┐
   │       GRU (32 units)         │  <-- Encodes high-level sequential dynamics
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Dense Head (64 -> 32 -> 7)  │  <-- Multi-step ReLU forecast projection
   └──────────────┬───────────────┘
                  │
                  ▼
   [Forecast Horizon: Next 7 Days]
```

### Key Technical Attributes
- **Input Window (Lookback):** 14-day sliding multivariate window.
- **Output Horizon:** 7-day simultaneous multi-step forecast vector.
- **Loss Function:** `Huber Loss` ($\delta = 1.0$), providing quadratic convergence for standard variations while resisting sudden sales spikes.
- **Optimizer:** `Adam` ($lr=10^{-3}$) paired with `ReduceLROnPlateau` and `EarlyStopping`.
- **Cyclical Encoding:** Continuous trigonometric harmonic projections ($\sin/\cos$) for day-of-week and month cycles.
- **Feature Set:** Past unit sales, ambient temperature, weather conditions, national holiday markers, weekend flags, promotional discounts, inflation index, and one-hot product identifiers.

---

## ✨ Features

- **Granular Target-Day Projections:** Pinpoint exact demand for specific operational days (e.g., *"How many units of Product X will sell next Tuesday?"*) along with 95% confidence intervals.
- **Automated Inventory Buffer Engine:**
  - **Safety Stock ($SS$):** Statistically scaled to a 95% service level ($Z = 1.65$).
  - **Reorder Point ($ROP$):** Dynamically computed based on lead time and variance.
- **Perishable Spoilage & Waste Mitigation:** Tracks physical shelf life (3–6 days) against projected consumption, alerting managers to dead stock and calculating revenue-maximizing discount markdowns.
- **"What-If" Scenario Simulation Studio:** Real-time stress-testing to assess the demand impact of extreme weather (e.g., storms, heatwaves), promotional flash sales, or holiday surges.
- **ERP & WMS SaaS Connector:** Clean JSON REST API ready for ingestion into SAP S/4HANA, Microsoft Dynamics 365, Logo Tiger, Netsis, Shopify, and NetSuite.
- **Modern Responsive Glassmorphic Dashboard:** Built with vanilla CSS and Chart.js for visualization of historical actuals, forecast curves, and loss metrics.

---

## 📐 Mathematical Formulations

### 1. Safety Stock ($SS$)
$$SS = Z \times \sigma_d \times \sqrt{L}$$
Where:
- $Z = 1.65$ (standard normal distribution quantile for a 95% cycle service level)
- $\sigma_d$ = standard deviation of daily sales over the trailing 30-day period
- $L$ = supplier replenishment lead time in days

### 2. Reorder Point ($ROP$)
$$ROP = (\bar{d} \times L) + SS$$
Where:
- $\bar{d}$ = average forecast daily demand over the replenishment horizon

### 3. Spoilage & Waste Risk Rule
$$\text{Surplus Units} = \max\left(0, \text{Stock}_{\text{current}} - \sum_{t=1}^{\text{ShelfLife}} \hat{y}_t\right)$$
If $\text{Surplus Units} > 0$, an alert triggers with an optimal clearance markdown recommendation before goods expire.

---

## 📁 Repository Structure

```
├── data/
│   ├── products_meta.json       # Product catalog, cost, price & lead-time metadata
│   └── retail_sales_data.csv    # Multi-year multivariate historical sales dataset
├── saved_model/
│   ├── best_demand_model.keras  # Checkpointed optimal model weights
│   ├── demand_forecast_model.keras # Production model bundle
│   ├── feat_scaler.joblib       # Fitted feature MinMaxScaler
│   ├── sales_scaler.joblib      # Fitted target MinMaxScaler
│   └── model_metrics.json       # Validation loss, MAE, RMSE & MAPE history
├── static/
│   ├── app.js                   # Client-side analytics, Chart.js integrations & API handlers
│   ├── index.html               # Responsive executive dashboard & scenario studio
│   └── style.css                # Dark-mode glassmorphic interface stylesheet
├── data_generator.py            # Synthetic multivariate retail time-series generator
├── inventory_engine.py          # Inventory optimization, ROP, safety stock & analytics
├── main.py                      # FastAPI REST server & background task worker
├── train_model.py               # TensorFlow Bi-LSTM + Multi-Head Attention training script
├── pyproject.toml               # Modern Python packaging configuration
├── requirements.txt             # Exact environment dependencies
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Git

### 2. Clone Repository & Setup Environment
```bash
git clone https://github.com/your-username/retail-demand-forecasting.git
cd retail-demand-forecasting

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate Data & Train Model (Optional)
The repository includes pre-generated data and trained weights in `saved_model/`. To regenerate data and retrain from scratch:
```bash
# 1. Generate multi-year time-series data
python data_generator.py

# 2. Train the TensorFlow Deep Learning Model
python train_model.py
```

### 4. Run the Production Application
Launch the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
- **Interactive Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Docs:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **ReDoc API Documentation:** [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)

---

## 📡 REST API Reference

### 1. Executive Analytics Overview
```http
GET /api/overview
```
Returns system-wide KPIs, inventory valuation, total financial exposure to stockouts, perishable waste risk, and item-by-item status.

### 2. Product 7-Day Forecast & Inventory Diagnosis
```http
GET /api/forecast/{product_id}
```
**Example:** `GET /api/forecast/PROD_001`

**Sample Response:**
```json
{
  "product": {
    "id": "PROD_001",
    "name": "Günlük Organik Süt (1L)",
    "category": "Süt & Şarküteri",
    "shelf_life_days": 4,
    "lead_time_days": 1,
    "unit_cost": 28.0,
    "selling_price": 42.0,
    "current_stock": 190
  },
  "next_tuesday_forecast": {
    "date": "2026-10-13",
    "day_name": "Salı",
    "predicted_sales": 118,
    "lower_bound": 104,
    "upper_bound": 133
  },
  "future_7days": [
    {
      "day_index": 1,
      "date": "2026-10-13",
      "day_name": "Salı",
      "predicted_sales": 118,
      "lower_bound": 104,
      "upper_bound": 133
    }
  ],
  "inventory_analytics": {
    "current_stock": 190,
    "total_7day_demand": 1030,
    "safety_stock": 46,
    "reorder_point": 193,
    "stockout_risk": true,
    "depletion_day": "Çarşamba",
    "lost_sales_units": 840,
    "lost_revenue_try": 35280.0,
    "recommended_reorder_qty": 886,
    "action_type": "STOCKOUT_ALERT"
  }
}
```

### 3. What-If Scenario Simulation
```http
POST /api/simulate
Content-Type: application/json

{
  "product_id": "PROD_001",
  "temperature": 12.0,
  "weather_code": 3,
  "discount_pct": 0.20,
  "is_holiday": 0
}
```

### 4. ERP & WMS Purchase Order Export
```http
GET /api/erp/export
```
Returns structured procurement recommendations for automated warehouse replenishment.

---

## 💼 ERP / WMS Monetization & Plugin Architecture

DemandFlow AI is designed to be packaged and sold as a **SaaS Plugin** for commercial ERP/WMS platforms:
- **Target Integrations:** SAP S/4HANA, Microsoft Dynamics, Logo Tiger, Netsis, Shopify Plus, NetSuite.
- **Billing Model:** Recurring monthly tier (e.g., $150–$350 / warehouse location).
- **Automation Pipeline:**
  1. Automated midnight cron synchronizes trailing 14-day transactional volumes.
  2. Model predicts future 7-day requirements across all SKUs.
  3. Automatic Purchase Orders (PO) are created for items where $\text{Current Stock} \le ROP$.
  4. Automatic promotional flash sales are dispatched to POS/e-commerce frontends for expiring inventory.

---

## 🛠️ Diagnostics & Benchmark Results

| Metric | Score | Description |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **11.28 units** | Average deviation in sales volume per product per day |
| **RMSE (Root Mean Squared Error)** | **14.83 units** | Penalizes severe prediction outliers |
| **MAPE** | **14.10%** | Mean absolute percentage error across test sequences |
| **Model Accuracy** | **85.90%** | Baseline generalized prediction reliability ($100 - \text{MAPE}$) |
| **Inference Latency** | **< 15 ms** | Sub-second real-time API response time per batch |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
