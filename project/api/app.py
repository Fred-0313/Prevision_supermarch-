"""
🚀 API FastAPI — Prédiction des ventes supermarché
Projet : Personne B — Jour 6

Endpoint : POST /predict
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os
from typing import Optional

# ─── Chargement des artefacts ────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_artifact(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artefact introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model_bundle = load_artifact("model.pkl")
    encoders     = load_artifact("encoders.pkl")
    MODEL        = model_bundle["model"]
    FEATURE_COLS = model_bundle["feature_cols"]
    METRICS      = model_bundle["metrics"]
    print("✅ Modèle chargé avec succès")
    print(f"   Métriques : MAE={METRICS['MAE']:,} | RMSE={METRICS['RMSE']:,} | R²={METRICS['R2']}")
except Exception as e:
    print(f"⚠️  Erreur de chargement : {e}")
    MODEL, FEATURE_COLS, METRICS, encoders = None, [], {}, {}


# ─── App FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="🛒 Supermarket Sales Forecast API",
    description=(
        "API de prédiction des ventes pour supermarchés béninois. "
        "Basée sur un modèle XGBoost optimisé — Projet Portfolio Data Science."
    ),
    version="1.0.0"
)


# ─── Schémas Pydantic ────────────────────────────────────────────────────────
class SalesPredictInput(BaseModel):
    # Identifiants du magasin
    Store_ID:    str  = Field(..., example="S001",    description="Identifiant du magasin (S001–S010)")
    Store_Type:  str  = Field(..., example="A",       description="Type de magasin : A, B ou C")
    City:        str  = Field(..., example="Cotonou", description="Ville du magasin")
    Store_Size:  int  = Field(..., example=4700,      description="Surface du magasin (m²)")
    Competitor_Distance: float = Field(..., example=2.5, description="Distance concurrent (km)")

    # Promotions & événements
    Promo:          int = Field(..., ge=0, le=1, example=1, description="Promotion active (0/1)")
    Holiday:        int = Field(..., ge=0, le=1, example=0, description="Jour férié (0/1)")
    School_Holiday: int = Field(..., ge=0, le=1, example=0, description="Congés scolaires (0/1)")
    Weekend:        int = Field(..., ge=0, le=1, example=1, description="Week-end (0/1)")

    # Météo
    Temperature: float = Field(..., example=28.5, description="Température (°C)")
    Rainfall:    float = Field(..., example=15.0, description="Pluviométrie (mm)")
    Fuel_Price:  float = Field(..., example=680.0, description="Prix carburant (FCFA/litre)")

    # Macro-économie
    Inflation_Rate:  float = Field(..., example=5.5,  description="Taux d'inflation (%)")
    Economic_Index:  float = Field(..., example=98.0, description="Indice économique")

    # Temporel
    month:        int = Field(..., ge=1,  le=12, example=6,  description="Mois (1–12)")
    day_of_week:  int = Field(..., ge=0,  le=6,  example=4,  description="Jour semaine (0=Lun, 6=Dim)")
    week_of_year: int = Field(..., ge=1,  le=53, example=24, description="Semaine de l'année")
    quarter:      int = Field(..., ge=1,  le=4,  example=2,  description="Trimestre (1–4)")
    season:       int = Field(..., ge=0,  le=3,  example=1,  description="Saison (0=Sèche, 1=P.Pluies, 2=P.Sèche, 3=G.Pluies)")
    year:         int = Field(..., example=2025, description="Année")


class SalesPredictOutput(BaseModel):
    Sales_prediction:  float
    Store_ID:          str
    City:              str
    Store_Type:        str
    confidence_note:   str
    model_metrics:     dict


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """Statut de l'API et informations sur le modèle."""
    return {
        "status": "✅ API opérationnelle",
        "model":  "XGBoost (optimisé)",
        "metrics": METRICS,
        "features": len(FEATURE_COLS),
        "docs":   "/docs"
    }


@app.post("/predict", response_model=SalesPredictOutput, tags=["Prédiction"])
def predict_sales(data: SalesPredictInput):
    """
    Prédit les ventes d'un magasin pour une journée donnée.

    **Input** : caractéristiques du magasin + conditions du jour
    **Output** : prédiction des ventes en FCFA
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Veuillez entraîner le modèle d'abord."
        )

    # 1. Encoder les variables catégorielles
    try:
        store_id_enc   = encoders["Store_ID"].transform([data.Store_ID])[0]
        store_type_enc = encoders["Store_Type"].transform([data.Store_Type])[0]
        city_enc       = encoders["City"].transform([data.City])[0]
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Valeur inconnue pour l'encodeur : {str(e)}. "
                   f"Store_ID valides : {list(encoders['Store_ID'].classes_)} | "
                   f"Store_Type valides : {list(encoders['Store_Type'].classes_)} | "
                   f"City valides : {list(encoders['City'].classes_)}"
        )

    # 2. Calculer les variables dérivées
    economic_pressure    = data.Inflation_Rate + (1 / max(data.Economic_Index, 0.01)) * 100
    promo_weekend        = data.Promo * data.Weekend
    promo_holiday        = data.Promo * data.Holiday
    promo_school_holiday = data.Promo * data.School_Holiday

    # 3. Construire le vecteur de features (même ordre que FEATURE_COLS)
    feature_values = {
        "Store_ID_enc":          store_id_enc,
        "Store_Type_enc":        store_type_enc,
        "City_enc":              city_enc,
        "Store_Size":            data.Store_Size,
        "Competitor_Distance":   data.Competitor_Distance,
        "Promo":                 data.Promo,
        "Holiday":               data.Holiday,
        "School_Holiday":        data.School_Holiday,
        "Weekend":               data.Weekend,
        "promo_weekend":         promo_weekend,
        "promo_holiday":         promo_holiday,
        "promo_school_holiday":  promo_school_holiday,
        "Temperature":           data.Temperature,
        "Rainfall":              data.Rainfall,
        "Fuel_Price":            data.Fuel_Price,
        "Inflation_Rate":        data.Inflation_Rate,
        "Economic_Index":        data.Economic_Index,
        "economic_pressure":     economic_pressure,
        "month":                 data.month,
        "day_of_week":           data.day_of_week,
        "week_of_year":          data.week_of_year,
        "quarter":               data.quarter,
        "season":                data.season,
        "year":                  data.year,
    }

    X = np.array([[feature_values[col] for col in FEATURE_COLS]])

    # 4. Prédiction
    prediction = float(MODEL.predict(X)[0])
    prediction = max(0.0, round(prediction, 2))  # pas de ventes négatives

    # 5. Note de confiance indicative (basée sur MAPE)
    mape = METRICS.get("MAPE(%)", 0)
    if mape < 5:
        note = f"Haute confiance (MAPE modèle : {mape:.2f}%)"
    elif mape < 10:
        note = f"Confiance modérée (MAPE modèle : {mape:.2f}%)"
    else:
        note = f"Confiance à vérifier (MAPE modèle : {mape:.2f}%)"

    return SalesPredictOutput(
        Sales_prediction = prediction,
        Store_ID         = data.Store_ID,
        City             = data.City,
        Store_Type       = data.Store_Type,
        confidence_note  = note,
        model_metrics    = METRICS
    )


@app.get("/stores", tags=["Référentiel"])
def get_valid_values():
    """Retourne les valeurs valides pour Store_ID, Store_Type, City."""
    if not encoders:
        return {"message": "Encoders non disponibles"}
    return {
        "Store_ID":    list(encoders["Store_ID"].classes_),
        "Store_Type":  list(encoders["Store_Type"].classes_),
        "City":        list(encoders["City"].classes_),
    }


# ─── Point d'entrée ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
