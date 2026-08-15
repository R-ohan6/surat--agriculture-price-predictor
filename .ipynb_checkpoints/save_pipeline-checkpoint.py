"""
save_pipeline.py

Run this ONCE, after training your model in the notebook, to bundle the
trained model + commodity encoder + historical price data into a single
SuratPricePredictor object that the Streamlit app can load directly.

Prerequisites (from the training notebook):
    - data/processed/surat_features_full.csv   (engineered features + Modal_Price)
    - A trained model whose feature columns match pipeline.FEATURE_ORDER
      plus one-hot commodity columns, e.g. the XGBoost model from the
      notebook's model comparison section.

If you're running this as a standalone script (not inside the notebook),
retrain a fresh model here for simplicity, matching the notebook's approach.
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from pipeline import SuratPricePredictor, FEATURE_ORDER
import os

# ---------- Load engineered features (output of the training notebook) ----------
engineered = pd.read_csv("data/processed/surat_model_ready.csv")
engineered["Arrival_Date"] = pd.to_datetime(engineered["Arrival_Date"])

# ---------- Rebuild the same train/test split and feature matrix as the notebook ----------
model_df = engineered[["Commodity", "Arrival_Date", "Modal_Price"] + FEATURE_ORDER].copy()
model_df = model_df.sort_values("Arrival_Date").reset_index(drop=True)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
commodity_encoded = encoder.fit_transform(model_df[["Commodity"]])
encoded_cols = encoder.get_feature_names_out(["Commodity"])
commodity_df = pd.DataFrame(commodity_encoded, columns=encoded_cols, index=model_df.index)

full_df = pd.concat([model_df.drop(columns=["Commodity"]), commodity_df], axis=1)

train_df = full_df[full_df["Arrival_Date"] < "2024-01-01"]
X_train = train_df.drop(columns=["Arrival_Date", "Modal_Price"])
y_train = train_df["Modal_Price"]

# ---------- Train the final model on all available training data ----------
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# ---------- Bundle everything into the pipeline object ----------
predictor = SuratPricePredictor()
predictor.fit(engineered, model)

os.makedirs("models", exist_ok=True)
predictor.save("models/surat_predictor.pkl")

print("Saved models/surat_predictor.pkl")
print("Available commodities:", predictor.available_commodities())

# Quick sanity check
from datetime import date
test_price = predictor.predict(
    commodity="Onion",
    target_date=date.today(),
    rainfall_mm=0.0,
    rainfall_7day_sum=0.0,
    temp_max=33.0,
    temp_min=25.0,
)
print(f"Sanity check prediction (Onion, today): Rs {test_price:.0f}")
