"""
main.py — FastAPI service for the Surat Vegetable Price Predictor

Wraps the SuratPricePredictor pipeline (see pipeline.py) behind a REST API,
so it can be called from a frontend (Streamlit, React, curl, etc.) without
needing Python/pandas/joblib installed on the client side.

Run locally with:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation
(FastAPI generates this automatically).

Expects:
    models/surat_predictor.pkl   (created by save_pipeline.py)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import date
from pipeline import SuratPricePredictor

app = FastAPI(
    title="Surat Vegetable Price Predictor API",
    description="Predicts fair-value wholesale vegetable prices for Surat APMC mandi, "
                "trained on 20 years of government market data, weather, and festival signals.",
    version="1.0.0",
)

# Allow the frontend (opened as a local file, or hosted separately) to call this API.
# For a portfolio project this is fine to leave open; for production you'd restrict
# allow_origins to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across requests
predictor: SuratPricePredictor = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = SuratPricePredictor.load("models/surat_predictor.pkl")


# ---------- Request / response schemas ----------

class PredictionRequest(BaseModel):
    commodity: str = Field(..., example="Onion")
    target_date: date = Field(..., example="2026-08-20")
    rainfall_mm: float = Field(0.0, ge=0, example=2.0, description="Today's rainfall in mm")
    rainfall_7day_sum: float = Field(0.0, ge=0, example=15.0, description="Cumulative rainfall over past 7 days in mm")
    temp_max: float = Field(..., example=33.0, description="Max temperature in Celsius")
    temp_min: float = Field(..., example=25.0, description="Min temperature in Celsius")


class PredictionResponse(BaseModel):
    commodity: str
    target_date: date
    predicted_price: float
    unit: str = "Rs per Quintal"
    last_reported_price: float | None = None


class HistoryPoint(BaseModel):
    date: date
    modal_price: float


# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "message": "Surat Vegetable Price Predictor API",
        "docs": "/docs",
        "commodities_endpoint": "/commodities",
        "predict_endpoint": "/predict (POST)",
    }


@app.get("/commodities", response_model=list[str])
def get_commodities():
    """List all commodities the model can predict for."""
    return predictor.available_commodities()


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    """Predict the fair-value price for a commodity on a given date."""
    if request.commodity not in predictor.available_commodities():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown commodity '{request.commodity}'. "
                   f"Valid options: {predictor.available_commodities()}",
        )

    try:
        prediction = predictor.predict(
            commodity=request.commodity,
            target_date=request.target_date,
            rainfall_mm=request.rainfall_mm,
            rainfall_7day_sum=request.rainfall_7day_sum,
            temp_max=request.temp_max,
            temp_min=request.temp_min,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    recent = predictor.recent_history(request.commodity, days=1)
    last_price = float(recent["Modal_Price"].iloc[-1]) if not recent.empty else None

    return PredictionResponse(
        commodity=request.commodity,
        target_date=request.target_date,
        predicted_price=round(prediction, 2),
        last_reported_price=last_price,
    )


@app.get("/history/{commodity}", response_model=list[HistoryPoint])
def get_history(commodity: str, days: int = 180):
    """Get recent historical prices for a commodity (for charting)."""
    if commodity not in predictor.available_commodities():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown commodity '{commodity}'. Valid options: {predictor.available_commodities()}",
        )

    recent = predictor.recent_history(commodity, days=days)
    return [
        HistoryPoint(date=row["Arrival_Date"].date(), modal_price=row["Modal_Price"])
        for _, row in recent.iterrows()
    ]
