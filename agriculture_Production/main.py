# -*- coding: utf-8 -*-
"""main.py

FastAPI backend for PS-12: Agricultural Yield Prediction.

Run locally:
    uvicorn main:app --reload
"""

import os
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Estimate crop yield (kg/ha) from region, crop group, year and last year's yield",
    version="1.0"
)

# Allow the frontend to call this API. On the server set ALLOWED_ORIGINS
# (comma separated) to restrict it, for example to the department website.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Load the saved model package (created at the end of the notebook)
package = joblib.load("best_yield_model.pkl")

model = package["model"]
scaler = package["scaler"]
feature_names = package["feature_names"]
numerical_features = package["numerical_features"]
numerical_medians = package["numerical_medians"]

FIRST_YEAR = package.get("first_year", 1962)

DISCLAIMER = (
    "Statistical estimate to support planning. "
    "It is not an official statistic."
)


class CropInput(BaseModel):
    region: Literal[
        "Africa", "Asia", "Europe",
        "North America", "South America", "Oceania"
    ]
    crop_group: Literal[
        "Cereals", "Vegetables & melons", "Fruits & nuts", "Oilseeds",
        "Roots & tubers", "Beverage & spice crops", "Pulses",
        "Sugar crops", "Fibre & other crops"
    ]
    year: int = Field(ge=1962, le=2050)
    yield_last_year: float = Field(gt=0, le=1000000)   # kg/ha


def build_features(crop: CropInput):
    """Same features as training. Only last year's yield is known,
    so the other history features take neutral values."""

    lag1 = float(np.log1p(crop.yield_last_year))

    values = {
        "lag1": lag1,
        "lag2": lag1,
        "lag3": lag1,
        "hist_mean": lag1,
        "roll_mean3": lag1,
        "roll_std3": 0.0,
        "delta_lag1": 0.0,
        "yield_trend3": 0.0,
        "area_lag1": float(numerical_medians["area_lag1"]),
        "area_growth": 0.0,
        "year_index": crop.year - FIRST_YEAR,
        "regional_shock": 0.0
    }

    # One-hot columns (the first category of each was dropped in training)
    row = {name: 0.0 for name in feature_names}

    for name in numerical_features:
        row[name] = values[name]

    for dummy in (f"Region_{crop.region}", f"Crop_group_{crop.crop_group}"):
        if dummy in row:
            row[dummy] = 1.0

    return pd.DataFrame([row])[feature_names]


def error_range(crop: CropInput):
    """Typical error (10th - 90th percentile) measured on the test data."""

    overall = package["error_range_overall"]

    by_group = package["error_range_by_crop_group"].get(crop.crop_group, overall)
    by_region = package["error_range_by_region"].get(crop.region, overall)

    # use the wider (more careful) of the two ranges
    low = min(by_group["low"], by_region["low"])
    high = max(by_group["high"], by_region["high"])

    return float(low), float(high)


@app.get("/")
def home():
    return {
        "message": "Crop Yield Prediction API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/model-info")
def model_info():
    m = package["test_metrics"]

    return {
        "model": package["best_model_name"],
        "target": "Crop yield in kg/ha",
        "trained_on": f"FAOSTAT crop data {FIRST_YEAR}-{package['last_year']}",
        "test_metrics": {
            "MAE_kg_per_ha": round(m["MAE"], 2),
            "MSE": round(m["MSE"], 2),
            "RMSE_kg_per_ha": round(m["RMSE"], 2),
            "R2": round(m["R2"], 4)
        }
    }


@app.post("/predict")
def predict_yield(crop: CropInput):

    features = build_features(crop)

    scaled = pd.DataFrame(
        scaler.transform(features),
        columns=feature_names
    )

    # model predicts log(1 + yield) -> convert back to kg/ha
    log_prediction = float(model.predict(scaled)[0])
    prediction = float(max(np.expm1(log_prediction), 0))

    low_gap, high_gap = error_range(crop)

    low = float(max(np.expm1(log_prediction + low_gap), 0))
    high = float(max(np.expm1(log_prediction + high_gap), 0))

    width = high_gap - low_gap

    if width < 0.32:
        uncertainty = "Low"
    elif width < 0.42:
        uncertainty = "Moderate"
    else:
        uncertainty = "High"

    change = (prediction - crop.yield_last_year) / crop.yield_last_year * 100

    notes: List[str] = [
        "Estimate uses only last year's yield, region, crop group and year."
    ]

    return {
        "year": crop.year,
        "predicted_yield_kg_per_ha": round(prediction, 1),
        "predicted_yield_tonnes_per_ha": round(prediction / 1000, 3),
        "last_year_yield_kg_per_ha": crop.yield_last_year,
        "change_vs_last_year_percent": round(change, 2),
        "likely_range_kg_per_ha": {
            "low": round(low, 1),
            "high": round(high, 1)
        },
        "uncertainty_level": uncertainty,
        "notes": notes,
        "disclaimer": DISCLAIMER
    }
