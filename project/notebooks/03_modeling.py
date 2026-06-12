# %% [markdown]
# # 🤖 Jour 3–5 : Modélisation — Personne B
# **Projet : Prédiction des ventes supermarché (Bénin)**
#
# Modèles entraînés :
# - 📏 Régression Linéaire (baseline)
# - 🌳 Decision Tree
# - 🌲 Random Forest
# - ⚡ XGBoost
#
# Métriques : MAE, RMSE, R²

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

MODELS_PATH = "../models/"
REPORTS_PATH = "../reports/"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# %% [markdown]
# ## 1. Chargement des données préparées

# %%
df = pd.read_csv("../data/data_prepared.csv", parse_dates=["Date"])

# Recréer le split temporel
df_train = df[df["year"] <= 2024].copy()
df_test  = df[df["year"] == 2025].copy()

# Charger la liste officielle des features
with open(os.path.join(MODELS_PATH, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

TARGET_COL = "Sales"

X_train = df_train[FEATURE_COLS]
y_train = df_train[TARGET_COL]
X_test  = df_test[FEATURE_COLS]
y_test  = df_test[TARGET_COL]

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Features ({len(FEATURE_COLS)}) : {FEATURE_COLS}")

# %% [markdown]
# ## 2. Fonction d'évaluation

# %%
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Entraîne et évalue un modèle, retourne un dict de métriques."""
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = root_mean_squared_error(y_te, preds)
    r2   = r2_score(y_te, preds)
    mape = np.mean(np.abs((y_te - preds) / y_te.clip(lower=1))) * 100

    print(f"{'─'*50}")
    print(f"  {name}")
    print(f"  MAE  : {mae:,.2f}  |  RMSE : {rmse:,.2f}")
    print(f"  R²   : {r2:.4f}  |  MAPE : {mape:.2f}%")
    return {
        "Modèle": name, "MAE": round(mae, 2),
        "RMSE": round(rmse, 2), "R2": round(r2, 4),
        "MAPE(%)": round(mape, 2), "model_obj": model, "preds": preds
    }

# %% [markdown]
# ## 3. 📏 Baseline — Régression Linéaire (Jour 3)

# %%
lr_results = evaluate_model(
    "Régression Linéaire (baseline)",
    LinearRegression(),
    X_train, y_train, X_test, y_test
)

# %% [markdown]
# ## 4. 🌳 Decision Tree (Jour 4)

# %%
dt_results = evaluate_model(
    "Decision Tree",
    DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42),
    X_train, y_train, X_test, y_test
)

# %% [markdown]
# ## 5. 🌲 Random Forest (Jour 4)

# %%
rf_results = evaluate_model(
    "Random Forest",
    RandomForestRegressor(n_estimators=200, max_depth=12,
                          min_samples_leaf=10, n_jobs=-1, random_state=42),
    X_train, y_train, X_test, y_test
)

# %% [markdown]
# ## 6. ⚡ XGBoost (Jour 4)

# %%
xgb_results = evaluate_model(
    "XGBoost",
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 n_jobs=-1, random_state=42, verbosity=0),
    X_train, y_train, X_test, y_test
)

# %% [markdown]
# ## 7. 📊 Comparaison des modèles

# %%
all_results = [lr_results, dt_results, rf_results, xgb_results]
results_df  = pd.DataFrame([{k: v for k, v in r.items()
                              if k not in ["model_obj", "preds"]}
                             for r in all_results])

print("\n📊 TABLEAU COMPARATIF")
print("="*60)
print(results_df.to_string(index=False))

# Graphique comparatif MAE / RMSE
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ["#94A3B8", "#F59E0B", "#10B981", "#2563EB"]
models_names = results_df["Modèle"].tolist()

for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    bars = ax.bar(models_names, results_df[metric], color=colors, edgecolor="white")
    ax.set_title(f"{metric} par modèle", fontweight="bold")
    ax.set_xticklabels(models_names, rotation=20, ha="right", fontsize=8)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{val:.3f}" if metric == "R2" else f"{val:,.0f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Graphique sauvegardé → reports/model_comparison.png")

# %% [markdown]
# ## 8. ⚙️ Optimisation des hyperparamètres (Jour 5)

# %%
# Optimisation ciblée sur XGBoost (meilleur candidat)
param_grid = {
    "n_estimators":    [200, 300, 400],
    "max_depth":       [4, 6, 8],
    "learning_rate":   [0.03, 0.05, 0.1],
    "subsample":       [0.7, 0.8],
    "colsample_bytree":[0.7, 0.8]
}

print("🔍 GridSearchCV en cours (XGBoost)...")
xgb_grid = GridSearchCV(
    XGBRegressor(n_jobs=-1, random_state=42, verbosity=0),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)
print(f"\n✅ Meilleurs paramètres : {xgb_grid.best_params_}")
print(f"   MAE CV : {-xgb_grid.best_score_:,.2f}")

# Évaluation du modèle optimisé
best_xgb = xgb_grid.best_estimator_
best_results = evaluate_model(
    "XGBoost (optimisé)",
    best_xgb,
    X_train, y_train, X_test, y_test
)

# %% [markdown]
# ## 9. 📈 Analyse des résidus — Modèle final

# %%
best_preds  = best_results["preds"]
residuals   = y_test.values - best_preds

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Analyse des résidus — XGBoost (optimisé)", fontsize=13, fontweight="bold")

# Résidus vs Prédictions
axes[0].scatter(best_preds, residuals, alpha=0.3, color="#2563EB", s=20)
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Valeurs prédites")
axes[0].set_ylabel("Résidus")
axes[0].set_title("Résidus vs Prédictions")

# Distribution des résidus
axes[1].hist(residuals, bins=40, color="#10B981", alpha=0.8, edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Distribution des résidus")
axes[1].set_xlabel("Résidu")

# Prédictions vs Réelles
axes[2].scatter(y_test, best_preds, alpha=0.3, color="#F59E0B", s=20)
min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())
axes[2].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
axes[2].set_xlabel("Valeurs réelles")
axes[2].set_ylabel("Valeurs prédites")
axes[2].set_title("Réel vs Prédit")

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, "residuals_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. 🎯 Feature Importance

# %%
feature_imp = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": best_xgb.feature_importances_
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(feature_imp["Feature"][:15], feature_imp["Importance"][:15],
               color="#2563EB", alpha=0.85)
ax.set_title("Top 15 Features — XGBoost (optimisé)", fontweight="bold", fontsize=13)
ax.set_xlabel("Importance (gain)")
ax.invert_yaxis()
for bar, val in zip(bars, feature_imp["Importance"][:15]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\n🏆 Top 10 features :")
print(feature_imp.head(10).to_string(index=False))

# Partager avec Personne A
feature_imp.to_csv(os.path.join(REPORTS_PATH, "feature_importance.csv"), index=False)
print("✅ Feature importance exportée → reports/feature_importance.csv")

# %% [markdown]
# ## 11. 💾 Sauvegarde du modèle final (Jour 5)

# %%
MODEL_SAVE = {
    "model":        best_xgb,
    "feature_cols": FEATURE_COLS,
    "best_params":  xgb_grid.best_params_,
    "metrics": {
        "MAE":  best_results["MAE"],
        "RMSE": best_results["RMSE"],
        "R2":   best_results["R2"],
        "MAPE": best_results["MAPE(%)"]
    }
}

with open(os.path.join(MODELS_PATH, "model.pkl"), "wb") as f:
    pickle.dump(MODEL_SAVE, f)

print("✅ Modèle final sauvegardé → models/model.pkl")
print(f"\n📦 Contenu sauvegardé :")
print(f"   - model        : XGBRegressor (optimisé)")
print(f"   - feature_cols : {len(FEATURE_COLS)} features")
print(f"   - best_params  : {xgb_grid.best_params_}")
print(f"   - metrics      : MAE={best_results['MAE']:,} | RMSE={best_results['RMSE']:,} | R²={best_results['R2']}")

# %% [markdown]
# ## 12. 📋 Résumé final — Tableau de bord

# %%
print("\n" + "="*65)
print("   RÉSUMÉ FINAL — MODÉLISATION PERSONNE B")
print("="*65)
print(f"\n{'Modèle':<30} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE':>8}")
print("-"*65)
for r in all_results + [best_results]:
    print(f"{r['Modèle']:<30} {r['MAE']:>10,.0f} {r['RMSE']:>10,.0f} "
          f"{r['R2']:>8.4f} {r['MAPE(%)']:>7.2f}%")
print("="*65)
print(f"\n🏆 Modèle sélectionné : XGBoost (optimisé)")
print(f"   MAE  = {best_results['MAE']:,.2f} (erreur moyenne en unités de vente)")
print(f"   RMSE = {best_results['RMSE']:,.2f}")
print(f"   R²   = {best_results['R2']:.4f} ({best_results['R2']*100:.1f}% de variance expliquée)")
print(f"   MAPE = {best_results['MAPE(%)']:.2f}%")
