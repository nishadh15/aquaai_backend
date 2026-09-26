"""AquaAI API: ESP32 sensor ingestion and explainable water recommendations."""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import water_model


class SensorReading(BaseModel):
    moisture: float = Field(..., ge=0, le=100, description="Calibrated soil moisture percentage")
    temperature: float = Field(..., ge=-10, le=60, description="Air temperature in degrees Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity percentage")
    rain: bool = Field(False, description="True when rain is currently detected or reliably forecast")
    crop: Literal["Tomato", "Chili"] = "Tomato"
    growth_stage: Literal["seedling", "vegetative", "flowering", "fruiting"] = "vegetative"
    previous_water: float = Field(0, ge=0, le=5, description="Water applied today, in L/plant")


class StoredReading(SensorReading):
    timestamp: str
    recommended_water: float
    pump_status: str
    reason: str


def allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGINS", "*").strip()
    return [origin.strip() for origin in configured.split(",") if origin.strip()] or ["*"]


app = FastAPI(
    title="AquaAI API",
    version="1.0.0",
    description="Single-zone smart-irrigation prototype API using synthetic Random Forest training data.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def initial_reading() -> StoredReading:
    reading = SensorReading(
        moisture=38,
        temperature=30,
        humidity=58,
        rain=False,
        crop="Tomato",
        growth_stage="flowering",
        previous_water=0.3,
    )
    result = water_model.recommend(reading.model_dump())
    return StoredReading(
        **reading.model_dump(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        recommended_water=result.recommended_water,
        pump_status=result.pump_status,
        reason=result.reason,
    )


recent_readings: deque[StoredReading] = deque([initial_reading()], maxlen=12)


def enrich(reading: SensorReading) -> StoredReading:
    recommendation = water_model.recommend(reading.model_dump())
    return StoredReading(
        **reading.model_dump(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        recommended_water=recommendation.recommended_water,
        pump_status=recommendation.pump_status,
        reason=recommendation.reason,
    )


@app.get("/", tags=["status"])
def root() -> dict:
    return {"message": "AquaAI API is running. Open /docs for interactive API documentation."}


@app.get("/api/health", tags=["status"])
def health() -> dict:
    return {"status": "ok", "service": "AquaAI FastAPI backend", "model_ready": True}


@app.get("/api/model-info", tags=["model"])
def model_info() -> dict:
    return water_model.metadata()


@app.get("/api/synthetic-data", tags=["model"])
def synthetic_data(limit: int = 10) -> dict:
    """Return a small sample only; the training set is intentionally synthetic."""
    if not 1 <= limit <= 50:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 50")
    return {
        "metadata": water_model.metadata(),
        "sample": water_model.dataset.head(limit).to_dict(orient="records"),
    }


@app.post("/api/recommendation", response_model=StoredReading, tags=["irrigation"])
def recommendation(reading: SensorReading) -> StoredReading:
    """Get a recommendation without saving a reading to the dashboard history."""
    return enrich(reading)


@app.post("/api/sensor-data", response_model=StoredReading, status_code=201, tags=["irrigation"])
def receive_sensor_data(reading: SensorReading) -> StoredReading:
    """ESP32 or dashboard submits one calibrated reading; keep its recommendation in recent history."""
    saved = enrich(reading)
    recent_readings.appendleft(saved)
    return saved


@app.get("/api/recent-readings", response_model=list[StoredReading], tags=["dashboard"])
def get_recent_readings() -> list[StoredReading]:
    return list(recent_readings)


@app.get("/api/dashboard", tags=["dashboard"])
def dashboard() -> dict:
    latest = recent_readings[0]
    return {"latest": latest, "recent_readings": list(recent_readings), "storage": "in-memory prototype history"}
