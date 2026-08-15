"""
pipeline.py — Surat Vegetable Price Predictor: formal pipeline

Wraps feature engineering, the fitted commodity encoder, and the trained
model into a single class with a clean predict() interface, so training
and inference (the Streamlit app) always use identical logic.

Usage (training):
    from pipeline import SuratPricePredictor

    predictor = SuratPricePredictor()
    predictor.fit(engineered_df, model)   # engineered_df = output of feature engineering notebook section
    predictor.save("models/surat_predictor.pkl")

Usage (inference / app):
    predictor = SuratPricePredictor.load("models/surat_predictor.pkl")
    price = predictor.predict(
        commodity="Onion",
        target_date=date(2026, 8, 20),
        rainfall_mm=5.0,
        rainfall_7day_sum=20.0,
        temp_max=33.0,
        temp_min=25.0,
    )
"""

import pandas as pd
import numpy as np
import joblib
import holidays as holidays_lib
from datetime import date
from sklearn.preprocessing import OneHotEncoder

MAJOR_FESTIVALS = ["Diwali (Deepavali)", "Holi", "Dussehra"]

FEATURE_ORDER = [
    "rainfall_mm", "rainfall_7day_sum", "temp_max", "temp_min",
    "days_to_nearest_festival", "is_festival_week",
    "year", "month", "day_of_week", "week_of_year",
    "price_lag_1", "price_lag_7", "price_lag_14", "price_lag_30",
    "price_rolling_mean_7D", "price_rolling_std_7D",
    "price_rolling_mean_30D", "price_rolling_std_30D",
]


class SuratPricePredictor:
    """
    Bundles:
      - historical price data (for computing lag/rolling features at inference time)
      - a fitted OneHotEncoder for the Commodity field
      - the trained regression model
    into one object with a single predict() method, so the app never has to
    re-implement feature engineering separately from training.
    """

    def __init__(self):
        self.model = None
        self.encoder = None
        self.history = None  # DataFrame: Commodity, Arrival_Date, Modal_Price

    # ---------------- Training-time setup ----------------

    def fit(self, engineered_df: pd.DataFrame, model):
        """
        engineered_df: the feature-engineered dataframe produced during training
                       (must contain Commodity, Arrival_Date, Modal_Price).
        model: an already-trained regressor (e.g., fitted XGBRegressor) whose
               feature columns follow FEATURE_ORDER + one-hot commodity columns.
        """
        self.history = (
            engineered_df[["Commodity", "Arrival_Date", "Modal_Price"]]
            .dropna(subset=["Modal_Price"])
            .sort_values(["Commodity", "Arrival_Date"])
            .reset_index(drop=True)
        )

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(engineered_df[["Commodity"]])

        self.model = model
        return self

    # ---------------- Feature computation (shared logic) ----------------

    def _compute_recency_features(self, commodity: str, target_date: pd.Timestamp) -> dict:
        """Look up lag/rolling price features from history, as of target_date."""
        sub = self.history[
            (self.history["Commodity"] == commodity) & (self.history["Arrival_Date"] < target_date)
        ].tail(30)

        if sub.empty:
            return {k: np.nan for k in [
                "price_lag_1", "price_lag_7", "price_lag_14", "price_lag_30",
                "price_rolling_mean_7D", "price_rolling_std_7D",
                "price_rolling_mean_30D", "price_rolling_std_30D",
            ]}

        prices = sub["Modal_Price"]
        return {
            "price_lag_1": prices.iloc[-1] if len(prices) >= 1 else np.nan,
            "price_lag_7": prices.iloc[-7] if len(prices) >= 7 else prices.iloc[-1],
            "price_lag_14": prices.iloc[-14] if len(prices) >= 14 else prices.iloc[-1],
            "price_lag_30": prices.iloc[-30] if len(prices) >= 30 else prices.iloc[-1],
            "price_rolling_mean_7D": prices.tail(7).mean(),
            "price_rolling_std_7D": prices.tail(7).std(),
            "price_rolling_mean_30D": prices.tail(30).mean(),
            "price_rolling_std_30D": prices.tail(30).std(),
        }

    @staticmethod
    def _compute_festival_features(target_date: date) -> dict:
        india_holidays = holidays_lib.India(years=range(target_date.year - 1, target_date.year + 2))
        festival_dates = [d for d, name in india_holidays.items() if name in MAJOR_FESTIVALS]

        if festival_dates:
            days_to_nearest = min(abs((target_date - d).days) for d in festival_dates)
        else:
            days_to_nearest = 365

        return {
            "days_to_nearest_festival": days_to_nearest,
            "is_festival_week": days_to_nearest <= 7,
        }

    def build_feature_row(
        self, commodity: str, target_date: date,
        rainfall_mm: float, rainfall_7day_sum: float,
        temp_max: float, temp_min: float,
    ) -> pd.DataFrame:
        """Assemble a single-row, correctly-ordered feature dataframe ready for model.predict()."""
        target_ts = pd.Timestamp(target_date)

        row = {
            "rainfall_mm": rainfall_mm,
            "rainfall_7day_sum": rainfall_7day_sum,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "year": target_date.year,
            "month": target_date.month,
            "day_of_week": target_date.weekday(),
            "week_of_year": target_date.isocalendar()[1],
        }
        row.update(self._compute_festival_features(target_date))
        row.update(self._compute_recency_features(commodity, target_ts))

        base_df = pd.DataFrame([row])[FEATURE_ORDER]

        commodity_encoded = self.encoder.transform(pd.DataFrame([[commodity]], columns=["Commodity"]))
        commodity_df = pd.DataFrame(commodity_encoded, columns=self.encoder.get_feature_names_out(["Commodity"]))

        full_row = pd.concat([base_df, commodity_df], axis=1)

        # Align to the exact column order the model was trained on, if available
        if hasattr(self.model, "get_booster"):
            full_row = full_row[self.model.get_booster().feature_names]
        elif hasattr(self.model, "feature_names_in_"):
            full_row = full_row[self.model.feature_names_in_]

        return full_row

    # ---------------- Inference ----------------

    def predict(
        self, commodity: str, target_date: date,
        rainfall_mm: float, rainfall_7day_sum: float,
        temp_max: float, temp_min: float,
    ) -> float:
        features = self.build_feature_row(
            commodity, target_date, rainfall_mm, rainfall_7day_sum, temp_max, temp_min
        )
        return float(self.model.predict(features)[0])

    def recent_history(self, commodity: str, days: int = 180) -> pd.DataFrame:
        """Convenience method for the app to plot a recent trend chart."""
        return (
            self.history[self.history["Commodity"] == commodity]
            .sort_values("Arrival_Date")
            .tail(days)
        )

    def available_commodities(self):
        return sorted(self.history["Commodity"].unique())

    # ---------------- Persistence ----------------

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SuratPricePredictor":
        return joblib.load(path)
