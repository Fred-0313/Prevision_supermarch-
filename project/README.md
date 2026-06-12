# 🛒 Supermarket Sales Forecasting — Bénin

> Prédire les ventes journalières de supermarchés béninois pour améliorer les prévisions commerciales.

## 📌 Résultats clés

| Modèle | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Régression Linéaire (baseline) | 696 | 864 | 0.8923 | 7.63% |
| Decision Tree | 880 | 1 120 | 0.8190 | 9.53% |
| Random Forest | 790 | 1 004 | 0.8545 | 8.57% |
| **XGBoost ✅** | **696** | **874** | **0.8898** | **7.55%** |

**Modèle final : XGBoost** — sélectionné pour sa robustesse et son interprétabilité via feature importance.

## 📂 Structure du projet

```
project/
├── data/
│   ├── supermarket_sales_forecasting.csv   # Données brutes
│   └── data_prepared.csv                  # Données encodées + features
│
├── notebooks/
│   ├── 01_data_preparation.py             # Encodage, split, feature engineering
│   └── 03_modeling.py                     # Baseline → XGBoost + évaluation
│
├── models/
│   ├── model.pkl                          # XGBoost final + métriques
│   ├── encoders.pkl                       # LabelEncoders (Store_ID, Store_Type, City)
│   └── feature_cols.pkl                  # Liste officielle des features
│
├── api/
│   └── app.py                             # FastAPI — endpoint /predict
│
├── reports/
│   ├── model_comparison.png              # Comparaison MAE / RMSE / R²
│   ├── feature_importance.png            # Top 15 features XGBoost
│   ├── feature_importance.csv            # Export pour Personne A
│   └── residuals_analysis.png            # Analyse des résidus
│
├── requirements.txt
└── README.md
```

## 🔑 Top features (XGBoost)

1. **Store_Type_enc** (33.9%) — Le type de magasin est le premier déterminant
2. **Promo** (19.2%) — Les promotions ont un impact majeur
3. **Store_Size** (10.9%) — La surface explique une part structurelle
4. **promo_weekend** (9.3%) — L'interaction Promo × Weekend amplifie les effets
5. **Competitor_Distance** (3.8%) — La concurrence impacte modérément

## 🚀 Lancer l'API

```bash
# Installer les dépendances
pip install -r requirements.txt

# Entraîner d'abord le modèle
cd notebooks && python 01_data_preparation.py && python 03_modeling.py

# Lancer l'API
cd ../api && uvicorn app:app --reload --port 8000
```

Documentation interactive : http://localhost:8000/docs

### Exemple de requête

```json
POST /predict
{
  "Store_ID": "S001",
  "Store_Type": "A",
  "City": "Cotonou",
  "Store_Size": 4700,
  "Competitor_Distance": 2.5,
  "Promo": 1,
  "Holiday": 0,
  "School_Holiday": 0,
  "Weekend": 1,
  "Temperature": 29.0,
  "Rainfall": 10.0,
  "Fuel_Price": 680.0,
  "Inflation_Rate": 5.5,
  "Economic_Index": 98.0,
  "month": 6,
  "day_of_week": 5,
  "week_of_year": 24,
  "quarter": 2,
  "season": 1,
  "year": 2025
}
```

### Réponse

```json
{
  "Sales_prediction": 14823.50,
  "Store_ID": "S001",
  "City": "Cotonou",
  "Store_Type": "A",
  "confidence_note": "Haute confiance (MAPE modèle : 7.55%)",
  "model_metrics": {"MAE": 696, "RMSE": 874, "R2": 0.8898, "MAPE(%)": 7.55}
}
```

## 🔁 Synchronisation avec Personne A

### Après EDA (validation croisée)
- ✅ Variables retenues : toutes les 24 features listées ci-dessus
- ✅ Variables dérivées validées : `economic_pressure`, `promo_weekend`, `season`
- ⚠️ `Rainfall` et `Temperature` : corrélation faible — à discuter

### Après modélisation (partage résultats)
- Top features exportées → `reports/feature_importance.csv`
- Le `Store_Type` domine : les magasins A ont des ventes structurellement plus élevées
- La `Promo` a un effet fort mais **amplifié le week-end** (interaction)
- L'effet macroéconomique (`Economic_Index`, `Inflation_Rate`) reste modéré

## 📊 Dataset

- **Source** : supermarket_sales_forecasting.csv
- **Période** : 01/01/2023 → 31/12/2025
- **Observations** : 5 500 lignes
- **Magasins** : 10 (S001–S010) dans 10 villes du Bénin
- **Target** : Sales (ventes journalières en FCFA)
