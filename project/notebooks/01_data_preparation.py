# %% [markdown]
# # 📦 Jour 1–2 : Préparation des données — Personne B
# **Projet : Prédiction des ventes supermarché (Bénin)**
#
# Étapes couvertes :
# - Chargement et inspection
# - Encodage des variables catégorielles
# - Feature engineering (date)
# - Split Train / Test (temporel)
# - Vérification des distributions

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Chemins
DATA_PATH   = "../data/supermarket_sales_forecasting.csv"
MODELS_PATH = "../models/"

os.makedirs(MODELS_PATH, exist_ok=True)

# %% [markdown]
# ## 1. Chargement des données

# %%
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
print(f"Shape : {df.shape}")
print(f"Période : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"\nTypes :\n{df.dtypes}")
print(f"\nValeurs manquantes :\n{df.isnull().sum()}")
df.head()

# %% [markdown]
# ## 2. Feature Engineering — Variables temporelles

# %%
df["month"]        = df["Date"].dt.month
df["day_of_week"]  = df["Date"].dt.dayofweek          # 0=Lundi … 6=Dimanche
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
df["quarter"]      = df["Date"].dt.quarter
df["year"]         = df["Date"].dt.year

def get_season(month):
    """Saisons adaptées au Bénin (climat tropical)"""
    if month in [11, 12, 1, 2]:   return 0  # Saison sèche principale
    elif month in [3, 4, 5]:       return 1  # Petite saison des pluies
    elif month in [6, 7]:          return 2  # Petite saison sèche
    else:                          return 3  # Grande saison des pluies

df["season"] = df["month"].apply(get_season)

# Variable composite : pression économique
df["economic_pressure"] = df["Inflation_Rate"] + (1 / df["Economic_Index"].clip(lower=0.01)) * 100

# Interaction promotionnelle (synergies)
df["promo_weekend"]        = df["Promo"] * df["Weekend"]
df["promo_holiday"]        = df["Promo"] * df["Holiday"]
df["promo_school_holiday"] = df["Promo"] * df["School_Holiday"]

print("Features temporelles et interactions créées :")
print(df[["month","day_of_week","week_of_year","quarter","season",
          "economic_pressure","promo_weekend","promo_holiday"]].head())

# %% [markdown]
# ## 3. Encodage des variables catégorielles

# %%
# Colonnes catégorielles à encoder
cat_cols = ["Store_ID", "Store_Type", "City"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Sauvegarde des encoders (nécessaire pour l'API)
with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)
print("\n✅ Encoders sauvegardés → models/encoders.pkl")

# %% [markdown]
# ## 4. Sélection des features finales

# %%
FEATURE_COLS = [
    # Store features
    "Store_ID_enc", "Store_Type_enc", "City_enc",
    "Store_Size", "Competitor_Distance",
    # Promotions & événements
    "Promo", "Holiday", "School_Holiday", "Weekend",
    "promo_weekend", "promo_holiday", "promo_school_holiday",
    # Météo & carburant
    "Temperature", "Rainfall", "Fuel_Price",
    # Macro-économie
    "Inflation_Rate", "Economic_Index", "economic_pressure",
    # Temporel
    "month", "day_of_week", "week_of_year", "quarter", "season", "year"
]

TARGET_COL = "Sales"

print(f"Nombre de features : {len(FEATURE_COLS)}")
print(f"Target : {TARGET_COL}")

# %% [markdown]
# ## 5. Split Train / Test (temporel — respecte la chronologie)

# %%
# Split temporel : train 2023+2024, test 2025
df_train = df[df["year"] <= 2024].copy()
df_test  = df[df["year"] == 2025].copy()

X_train = df_train[FEATURE_COLS]
y_train = df_train[TARGET_COL]
X_test  = df_test[FEATURE_COLS]
y_test  = df_test[TARGET_COL]

print(f"Train : {X_train.shape[0]} lignes ({df_train['Date'].min().date()} → {df_train['Date'].max().date()})")
print(f"Test  : {X_test.shape[0]} lignes ({df_test['Date'].min().date()}  → {df_test['Date'].max().date()})")
print(f"Ratio : {len(X_test)/len(df)*100:.1f}% test")

# %% [markdown]
# ## 6. Vérification de la distribution Train vs Test

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(y_train, bins=40, color="#2563EB", alpha=0.7, label="Train")
axes[0].hist(y_test, bins=40, color="#F59E0B", alpha=0.7, label="Test")
axes[0].set_title("Distribution de Sales — Train vs Test")
axes[0].set_xlabel("Sales")
axes[0].legend()

# Boxplot par Store_Type
df_plot = df.copy()
df_plot["Split"] = df_plot["year"].apply(lambda y: "Train" if y <= 2024 else "Test")
df_plot.boxplot(column="Sales", by="Store_Type", ax=axes[1])
axes[1].set_title("Sales par Store_Type")
plt.suptitle("")
plt.tight_layout()
plt.savefig("../reports/distribution_train_test.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Graphique sauvegardé → reports/distribution_train_test.png")

# %% [markdown]
# ## 7. Sauvegarde des données préparées

# %%
# Sauvegarde pour reproductibilité
df.to_csv("../data/data_prepared.csv", index=False)

# Sauvegarde de la liste des features (utile pour l'API)
with open(os.path.join(MODELS_PATH, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("✅ Données préparées sauvegardées → data/data_prepared.csv")
print("✅ Feature columns sauvegardées  → models/feature_cols.pkl")
print()
print("📊 Résumé Train :")
print(y_train.describe().round(2))
print("\n📊 Résumé Test :")
print(y_test.describe().round(2))
