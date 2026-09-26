"""Synthetic training data and transparent Random Forest recommendation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


Crop = Literal["Tomato", "Chili"]
GrowthStage = Literal["seedling", "vegetative", "flowering", "fruiting"]

FEATURE_COLUMNS = [
    "moisture",
    "temperature",
    "humidity",
    "rain",
    "crop",
    "growth_stage",
    "previous_water",
]


def build_synthetic_dataset(rows: int = 360, seed: int = 42) -> pd.DataFrame:
    """Make a clearly labelled, repeatable prototype dataset.

    Values imitate a small garden, not a real field trial. `required_water` is in
    litres per plant per day and provides a learnable target for the demo model.
    """
    rng = np.random.default_rng(seed)
    crops = rng.choice(["Tomato", "Chili"], size=rows, p=[0.62, 0.38])
    stages = rng.choice(
        ["seedling", "vegetative", "flowering", "fruiting"],
        size=rows,
        p=[0.18, 0.34, 0.24, 0.24],
    )
    moisture = rng.uniform(18, 72, size=rows)
    temperature = rng.uniform(20, 38, size=rows)
    humidity = rng.uniform(35, 88, size=rows)
    rain = rng.binomial(1, 0.22, size=rows)
    previous_water = rng.uniform(0.0, 1.8, size=rows)

    stage_factor = pd.Series(stages).map(
        {"seedling": 0.45, "vegetative": 0.85, "flowering": 1.10, "fruiting": 1.25}
    ).to_numpy()
    crop_factor = np.where(crops == "Tomato", 1.10, 0.88)
    dryness = np.clip((62 - moisture) / 44, 0, 1.15)
    heat_factor = np.clip((temperature - 23) * 0.035, 0, 0.55)
    humidity_relief = np.clip((humidity - 50) * 0.008, 0, 0.25)
    target = (
        0.10
        + (1.10 * dryness * crop_factor * stage_factor)
        + heat_factor
        - humidity_relief
        - (rain * 0.75)
        - (previous_water * 0.25)
        + rng.normal(0, 0.06, size=rows)
    )
    target = np.clip(target, 0, 2.5).round(2)

    return pd.DataFrame(
        {
            "moisture": moisture.round(1),
            "temperature": temperature.round(1),
            "humidity": humidity.round(1),
            "rain": rain,
            "crop": crops,
            "growth_stage": stages,
            "previous_water": previous_water.round(2),
            "required_water": target,
        }
    )


@dataclass
class Recommendation:
    recommended_water: float
    pump_status: str
    reason: str


class WaterRecommendationModel:
    """Small, reproducible ML component suitable for an SIH prototype demo."""

    def __init__(self) -> None:
        self.dataset = build_synthetic_dataset()
        categorical = ["crop", "growth_stage"]
        numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
        preprocessor = ColumnTransformer(
            [
                ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("number", "passthrough", numeric),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "forest",
                    RandomForestRegressor(
                        n_estimators=120, max_depth=8, random_state=42, min_samples_leaf=2
                    ),
                ),
            ]
        )
        self.pipeline.fit(self.dataset[FEATURE_COLUMNS], self.dataset["required_water"])

    def recommend(self, reading: dict) -> Recommendation:
        """Predict a bounded recommendation and apply readable safety rules."""
        # Rain is deliberately a deterministic override: users can explain it to judges.
        if reading["rain"]:
            return Recommendation(
                recommended_water=0.0,
                pump_status="POSTPONED",
                reason="Rain is reported, so irrigation is postponed to avoid unnecessary watering.",
            )

        # A sufficiently moist root zone should not be watered even if the model predicts a small value.
        if reading["moisture"] >= 62:
            return Recommendation(
                recommended_water=0.0,
                pump_status="OFF",
                reason="Soil moisture is at or above the 62% prototype threshold.",
            )

        frame = pd.DataFrame([{column: reading[column] for column in FEATURE_COLUMNS}])
        predicted = float(self.pipeline.predict(frame)[0])
        capped = round(max(0.0, min(predicted, 2.5)), 2)
        if capped < 0.10:
            return Recommendation(
                recommended_water=0.0,
                pump_status="OFF",
                reason="The predicted requirement is below the 0.10 L/plant/day minimum action threshold.",
            )
        return Recommendation(
            recommended_water=capped,
            pump_status="ON (recommended)",
            reason="Dryness, crop, growth stage, temperature, humidity, and previous watering indicate irrigation is needed.",
        )

    def metadata(self) -> dict:
        return {
            "dataset_type": "synthetic prototype data",
            "rows": len(self.dataset),
            "columns": list(self.dataset.columns),
            "model": "RandomForestRegressor (120 trees, fixed random seed)",
            "unit": "litres per plant per day",
            "disclaimer": "This is a demonstration model, not a calibrated or field-validated irrigation schedule.",
        }


water_model = WaterRecommendationModel()
