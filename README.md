# 🛒 Prévision des Ventes - Supermarchés Béninois

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0.8898_R²-EB6626?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit)
![MAPE](https://img.shields.io/badge/MAPE-7.55%25-success?style=flat-square)
![MAE](https://img.shields.io/badge/MAE-696_FCFA-blue?style=flat-square)

> **5 500 jours de tickets de caisse de 10 supermarchés béninois transformés en une API qui prédit demain à 7.55% près.**

Le projet de prévision des ventes des supermarchés béninois présente un système complet de bout en bout qui transforme 5 500 jours de ventes de 10 magasins de Cotonou à Djougou en prédictions journalières à 7,55% d'erreur, grâce à un modèle XGBoost (R² 89%, MAE 696 FCFA) déployé via une API FastAPI et une interface Streamlit, et qui révèle que le type de magasin porte 33,9% de l'explication et que la promotion génère +27,9% de ventes, +42% le week-end.

---

## 📌 Sommaire
- [Problème](#-problème)
- [Dataset](#-dataset)
- [Étapes du projet](#-étapes-du-projet)
- [Résultats](#-résultats)
- [Insights Business](#-insights-business)
- [Stack & Architecture](#-stack--architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure](#-structure)
- [Auteurs](#-auteurs)

---

## 🎯 Problème

Les responsables de supermarchés au Bénin planifient stocks et promos à l'intuition. 5% d'erreur = rupture ou gaspillage. 

**Question :** Comment prédire les ventes d'un jour J à partir de ses caractéristiques (magasin, promo, météo, macro) ?

**Notre réponse :** Pas un notebook Kaggle. Un vrai produit : EDA rigoureuse + benchmark 4 modèles + API + UI.

## 📊 Dataset

- **Fichier :** `supermarket_sales_forecasting.csv`
- **Volume :** 5 500 lignes, 01/01/2023 → 31/12/2025
- **Cible :** `Sales` FCFA/jour — Moy 9 889, ET 2 588, Min 1 507, Max 18 551
- **Qualité :** 0 valeur manquante. Split 80/20 stratifié par Store_Type, seed 42
- **Villes :** Cotonou, Porto-Novo, Parakou, Ouidah, Abomey-Calavi, Natitingou, Kandi, Bohicon, Lokossa, Djougou

**16 variables brutes :**
`Store_ID, Store_Type (A/B/C), Store_Size, City, Competitor_Distance, Promo, Holiday, School_Holiday, Weekend, Temperature, Rainfall, Fuel_Price, Inflation_Rate, Economic_Index, Date, Sales`

**24 features finales** après engineering : `month, day_of_week, week_of_year, quarter, season, promo_weekend, promo_holiday, promo_school_holiday, economic_pressure` + encodages.

## 🔬 Étapes du projet

| Étape | Qui | Quoi |
| :--- | :--- | :--- |
| **S1 - EDA** | BIAOU Sherif | Stats, corrélations, boxplots. Découverte : Promo×Weekend non additif. Store_Size +0.684, Competitor -0.475 |
| **S2 - Features** | Commun | Date → 5 temporelles, 4 interactions, LabelEncoder Store_ID/Type/City |
| **S2 - Test V2** | Commun | Retrait Temp/Fuel/year → MAE 696→703 (+1%). Décision : on garde V1 complet |
| **S3-S4 - Modélisation** | SOSSA Fred | 4 modèles, 5-Fold CV, MAE/RMSE/R²/MAPE |
| **S5-S6 - Déploiement** | SOSSA Fred | joblib, FastAPI 3 endpoints, Streamlit 3 onglets, découplage via requests |
| **S7-S8 - Doc** | Commun | sync_report, insights.md, README, rapport portfolio |

## 📈 Résultats

Test set : 1 100 lignes

| Modèle | MAE | RMSE | R² | MAPE |
| :--- | :--- | :--- | :--- | :--- |
| Régression Linéaire | 696 | 864 | **0.8923** | 7.63% |
| Decision Tree | 880 | 1120 | 0.8190 | 9.53% |
| Random Forest | 790 | 1004 | 0.8545 | 8.57% |
| **XGBoost (final)** | **696** | **874** | **0.8898** | **7.55%** |

**Choix XGBoost :** Pas le meilleur R² (Linéaire 0.8923), mais plus stable en CV, résidus normaux, capture interactions non-linéaires Promo×Type. 7.55% = standard retail "Bon" (5-10% = prod ready).

**Top Features :**
1. `Store_Type_enc` 33.86% - Structurel #1
2. `Promo` 19.20% - Levier actionnable #1, +2 543 FCFA/j
3. `Store_Size` 10.87%
4. `promo_weekend` 9.25% - Preuve interaction
5. `Competitor_Distance` 3.8%

Météo <0.5%, Macro <1.5% → inutile d'ajuster stocks sur météo au Bénin.

![Feature Importance](assets/feature_importance.png)
![Model Comparison](assets/model_comparison.png)
![Residuals](assets/residuals_analysis.png)

## 💡 Insights Business

Dashboard complet : `sync_report_A_B.png`

1.  **Type A = +58% vs Type C** (12 309 vs 7 804 FCFA/j) → Objectifs différenciés
2.  **Promo = +27.9%** (9 120 → 11 663) → ROI mesurable
3.  **Promo Week-end = +42%** (8 803→12 530) vs +28% semaine → **Reco #1 : concentrer budget Vend-Sam**
4.  **Jour férié = +21.4%** (9 554→11 600) → Anticiper appros
5.  **Villes :** Ouidah & Cotonou >12 500 (top), Djougou ~7 200 (booster promo)

## 🛠️ Stack & Architecture

**Stack :** Python 3.10, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, FastAPI, Pydantic, Uvicorn, Streamlit

**Architecture :**
```
Streamlit (8501) --requests--> FastAPI (8000) --> model.pkl + encoders.pkl
     EDA / Predict / Insights      / /stores /predict
```

**Artefacts :** `models/model.pkl`, `models/encoders.pkl`, `models/feature_cols.pkl` (24 cols officielles)

## 🚀 Installation

```bash
git clone https://github.com/votre-username/supermarket-sales-forecasting.git
cd supermarket-sales-forecasting

# Environnement
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Données
# Placer supermarket_sales_forecasting.csv dans data/
```

## ▶️ Utilisation

**1. Entraînement (si besoin)**
```bash
python scripts/01_data_preparation.py
python scripts/03_modeling.py
```

**2. Lancer l'API**
```bash
cd api
uvicorn app:app --reload --port 8000
# Docs : http://localhost:8000/docs
```

**3. Lancer l'UI**
```bash
streamlit run ui/stream_EDA.py --server.port 8501
# UI : http://localhost:8501
```

**Exemple requête API :**
```json
POST /predict
{
  "Store_ID": "S001",
  "Store_Type": "A",
  "Store_Size": 2500,
  "City": "Cotonou",
  "Competitor_Distance": 500,
  "Promo": 1,
  "Holiday": 0,
  "School_Holiday": 0,
  "Weekend": 1,
  "Temperature": 28.5,
  "Rainfall": 0,
  "Fuel_Price": 650,
  "Inflation_Rate": 3.2,
  "Economic_Index": 105,
  "month": 6,
  "day_of_week": 5
}
```
```json
Response: { "Sales_prediction": 14823.5, "confidence": "7.55%" }
```

## 📁 Structure
```
.
├── data/
│   └── supermarket_sales_forecasting.csv
├── models/
│   ├── model.pkl
│   ├── encoders.pkl
│   └── feature_cols.pkl
├── api/
│   └── app.py (FastAPI)
├── ui/
│   └── stream_EDA.py (Streamlit)
├── assets/
│   ├── distribution_train_test.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── residuals_analysis.png
│   └── sync_report_A_B.png
├── docs/
│   ├── insights.md
│   └── Rapport_Portfolio.pdf
├── requirements.txt
└── README.md
```

## ⚠️ Limites & Next Steps

**Limites honnêtes :** Dataset simulé, 11% variance inexpliquée, pas de cannibalisation inter-magasins, pas de coûts/marge (CA seulement), pas de monitoring drift, year retiré.

**Next :** Docker, CI/CD, Evidently/MLflow pour drift, LightGBM + quantile, Google Trends, coûts pour ROI promo réel.

## 👥 Auteurs

**BIAOU Sherif** — Data Analyst | EDA, Feature Engineering, Insights, Streamlit
- LinkedIn: linkedin.com/in/sherif-biaou

**SOSSA Fortuné Fred** — ML Engineer | Modélisation, Benchmark, FastAPI, MLOps
- LinkedIn: linkedin.com/in/fortune-sossa

Projet Personnel Collaboratif — juillet 2026 

---

