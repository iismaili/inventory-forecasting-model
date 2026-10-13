"""
FastAPI Demand Forecasting & Inventory Analytics Server.

Exposes RESTful endpoints for time-series predictions, what-if scenario simulations,
model diagnostics, model retraining, and enterprise ERP/WMS data integrations.
"""

import os
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inventory_engine import InventoryForecastingEngine

app = FastAPI(
    title="DemandFlow AI Engine",
    description="Enterprise Time-Series Forecasting & Inventory Optimization Engine for ERP/WMS",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

engine = InventoryForecastingEngine()


# Request / Response Schemas
class ScenarioRequest(BaseModel):
    product_id: str = Field(..., description="Unique product identifier (e.g. PROD_001)")
    temperature: float = Field(20.0, description="Ambient or regional temperature in Celsius")
    weather_code: int = Field(0, description="Categorical weather index (0: Sunny, 1: Cloudy, 3: Rain)")
    discount_pct: float = Field(0.0, ge=0.0, le=0.50, description="Promotional discount fraction (0.0 to 0.50)")
    is_holiday: int = Field(0, ge=0, le=1, description="Binary flag indicating whether date is a holiday")


class RetrainRequest(BaseModel):
    epochs: int = Field(25, ge=5, le=100, description="Training epochs for model fine-tuning")


training_state: Dict[str, Any] = {
    "is_training": False,
    "last_trained": "System initialization",
    "message": "Model is active and ready for inference.",
}


@app.get("/api/overview", tags=["Analytics"])
def get_overview() -> Dict[str, Any]:
    """Retrieve executive KPI indicators and risk summary for all catalog items."""
    try:
        return engine.get_executive_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/products", tags=["Catalog"])
def get_products() -> List[Dict[str, Any]]:
    """Retrieve the active catalog list with product master metadata."""
    return engine.get_products()


@app.get("/api/forecast/{product_id}", tags=["Forecasting"])
def get_forecast(product_id: str) -> Dict[str, Any]:
    """Retrieve baseline 7-day multi-step forecasts and inventory risks for a product."""
    try:
        return engine.predict_product_forecast(product_id)
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/simulate", tags=["What-If Simulation"])
def run_simulation(req: ScenarioRequest) -> Dict[str, Any]:
    """Simulate demand elasticity under hypothetical weather, pricing, or holiday conditions."""
    try:
        scenario = {
            "temperature": req.temperature,
            "weather_code": req.weather_code,
            "discount_pct": req.discount_pct,
            "is_holiday": req.is_holiday,
        }
        res = engine.predict_product_forecast(req.product_id, scenario=scenario)
        baseline = engine.predict_product_forecast(req.product_id)

        sim_tue = res["next_tuesday_forecast"]["predicted_sales"]
        base_tue = baseline["next_tuesday_forecast"]["predicted_sales"]
        tue_delta_pct = round(((sim_tue - base_tue) / max(1, base_tue)) * 100.0, 1)

        sim_total = res["inventory_analytics"]["total_7day_demand"]
        base_total = baseline["inventory_analytics"]["total_7day_demand"]
        total_delta_pct = round(((sim_total - base_total) / max(1, base_total)) * 100.0, 1)

        res["simulation_comparison"] = {
            "baseline_tuesday_sales": base_tue,
            "simulated_tuesday_sales": sim_tue,
            "tuesday_delta_pct": tue_delta_pct,
            "baseline_7day_total": base_total,
            "simulated_7day_total": sim_total,
            "total_delta_pct": total_delta_pct,
            "impact_summary": f"Senaryo etkisiyle Gelecek Salı talebi %{tue_delta_pct:+g} ({sim_tue - base_tue:+d} adet) değişti.",
        }
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/model-info", tags=["Diagnostics"])
def get_model_info() -> Dict[str, Any]:
    """Retrieve model architecture details, features, and historical evaluation metrics."""
    metrics = engine.metrics
    return {
        "framework": "TensorFlow 2.22.0 / Keras 3",
        "architecture": "Bidirectional LSTM (64 units) + Multi-Head Temporal Attention (4 heads, 32 dim) + GRU (32 units) + Dense Head",
        "input_features": metrics.get("feature_names", []),
        "sequence_length_days": metrics.get("sequence_length", 14),
        "forecast_horizon_days": metrics.get("forecast_horizon", 7),
        "metrics": metrics.get("metrics", {}),
        "loss_history": metrics.get("loss", []),
        "val_loss_history": metrics.get("val_loss", []),
        "mae_history": metrics.get("mae", []),
        "val_mae_history": metrics.get("val_mae", []),
        "training_status": training_state,
    }


def _execute_retraining_task(epochs: int) -> None:
    """Asynchronous background worker for model retraining."""
    global training_state
    training_state["is_training"] = True
    training_state["message"] = "Model retraining in progress on TensorFlow backend..."
    try:
        from train_model import train_and_evaluate

        metrics = train_and_evaluate(epochs=epochs)
        engine.load_assets()
        training_state["is_training"] = False
        training_state["last_trained"] = "Recently completed"
        training_state["message"] = f"Training finished successfully. Validation MAE: {metrics['metrics']['mae_units']} units."
    except Exception as exc:
        training_state["is_training"] = False
        training_state["message"] = f"Training error encountered: {str(exc)}"


@app.post("/api/retrain", tags=["Model Lifecycle"])
def retrain_model(req: RetrainRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Trigger background retraining of the deep learning model."""
    if training_state["is_training"]:
        return {"status": "busy", "message": "A training process is already running."}
    background_tasks.add_task(_execute_retraining_task, req.epochs)
    return {"status": "started", "message": f"Retraining task scheduled for {req.epochs} epochs."}


@app.get("/api/erp/export", tags=["ERP/WMS Integration"])
def erp_export() -> Dict[str, Any]:
    """
    Export automated procurement recommendations formatted for ERP/WMS systems
    (SAP S/4HANA, Microsoft Dynamics, Logo Tiger, Netsis, Shopify, Netsuite).
    """
    overview = engine.get_executive_overview()
    return {
        "erp_plugin_version": "v1.4.2",
        "timestamp": "2026-10-13T12:00:00Z",
        "currency": "TRY",
        "purchase_order_recommendations": [
            {
                "product_code": p["id"],
                "product_name": p["name"],
                "warehouse_stock": p["current_stock"],
                "forecast_next_tuesday": p["next_tuesday_sales"],
                "reorder_point": p["reorder_point"],
                "action": p["action_type"],
                "urgent_reorder_required": p["stockout_risk"],
            }
            for p in overview["products_summary"]
        ],
        "financial_exposure": {
            "spoilage_waste_risk_try": overview["kpis"]["total_waste_risk_try"],
            "stockout_lost_revenue_try": overview["kpis"]["total_lost_revenue_risk_try"],
        },
    }


# Static assets & frontend routing
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the single-page application dashboard."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
