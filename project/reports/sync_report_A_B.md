# Rapport de synchronisation — Personne A × Personne B
## Point 1 : Validation post-EDA

**Date :** Semaine 1 — Fin Jour 2

---

## Ce que l'EDA de Personne A confirme

### Variables fortement utiles (à conserver — confirmé par les deux côtés)

| Variable | Corrélation (A) | Importance XGBoost (B) | Statut |
|---|---|---|---|
| Store_Size | +0.684 | 10.9% (rang 3) | ✅ Garder |
| Store_Type | — (catégorielle) | **33.9% (rang 1)** | ✅ Garder — variable la plus importante |
| Promo | +0.451 | 19.2% (rang 2) | ✅ Garder — effet +27.9% sur les ventes |
| Competitor_Distance | -0.475 | 3.8% (rang 5) | ✅ Garder |
| Weekend | +0.209 | 2.9% | ✅ Garder |
| Holiday | +0.135 | 2.6% | ✅ Garder |
| School_Holiday | +0.114 | 1.2% | ✅ Garder |
| Economic_Index | +0.156 | 0.7% | ✅ Garder (signal macro) |
| Inflation_Rate | -0.130 | 0.4% | ✅ Garder (via economic_pressure) |

### Variables faibles (à surveiller)

| Variable | Corrélation (A) | Importance XGBoost (B) | Décision |
|---|---|---|---|
| Temperature | -0.026 | 0.196% (rang 23) | ⚠️ Retirée en V2 |
| Fuel_Price | -0.099 | 0.321% (rang 22) | ⚠️ Retirée en V2 |
| year | — | 0.179% (rang 24) | ⚠️ Retirée (pas de généralisation) |

**Résultat du test V2 :** retrait des 3 variables faibles → MAE passe de 696 à 703
→ **Décision : modèle V1 conservé** (les 24 features). La perte est marginale (~1%) mais le gain de simplicité ne justifie pas.

---

## Interactions validées

L'EDA de Personne A confirme deux interactions clés déjà intégrées dans le modèle B :

### Promo × Weekend
| | Semaine | Week-end |
|---|---|---|
| **Sans promo** | 8 803 | 9 962 |
| **Avec promo** | 11 316 | **12 530** |

→ L'effet Promo est **amplifié le week-end** (+41% vs semaine sans promo).
→ Variable `promo_weekend` déjà présente → importance 9.3% (rang 4). ✅

### Promo × Store_Type
| | Type A | Type B | Type C |
|---|---|---|---|
| **Sans promo** | 11 507 | 8 952 | 7 057 |
| **Avec promo** | **14 209** | 11 411 | 9 575 |

→ Le Store_Type amplifie l'effet promo de façon non linéaire.
→ XGBoost capture naturellement cette interaction via le Store_Type_enc (rang 1). ✅

---

## Insights business identifiés (pour README)

### 1. Le type de magasin structure les ventes (33.9% d'importance)
- Type A : 12 309 FCFA/jour en moyenne
- Type B : 9 718 FCFA/jour
- Type C : 7 804 FCFA/jour
- **Implication** : les décisions d'assortiment et de politique promotionnelle doivent être différenciées par type.

### 2. La promo est le levier opérationnel le plus fort
- +27.9% de ventes avec promotion active
- L'effet est **multiplicatif avec le type de magasin** : une promo en magasin A génère +2 700 FCFA/jour supplémentaires vs type C

### 3. La concurrence pèse structurellement (-0.47)
- Competitor_Distance est négativement corrélé : plus le concurrent est proche, plus les ventes baissent
- Ce n'est pas actionnable à court terme mais important pour la localisation future

### 4. L'économie macro a un effet réel mais modéré
- Economic_Index : +0.16 → un bon climat économique booste légèrement les ventes
- Inflation_Rate : -0.13 → inflation élevée comprime les achats

### 5. La saisonnalité est bien captée par le modèle
- Les variables `season`, `month`, `quarter` cumulent ~7% d'importance
- Le jour de la semaine a aussi un effet (day_of_week : 0.4%, mais amplifié via les interactions)

---

## Performances finales du modèle

| Modèle | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Régression Linéaire | 696 | 864 | 0.892 | 7.63% |
| Decision Tree | 880 | 1 120 | 0.819 | 9.53% |
| Random Forest | 790 | 1 004 | 0.855 | 8.57% |
| **XGBoost final** | **696** | **874** | **0.890** | **7.55%** |

**Interprétation :** Le modèle prédit les ventes avec une erreur moyenne de ~696 FCFA.
Sur une vente moyenne de ~9 890 FCFA, cela représente une précision de ~92.5%.

---

## Prochaine étape

- [x] EDA validée par Personne B
- [x] Features validées mutuellement  
- [x] Feature importance partagée avec Personne A (`reports/feature_importance.csv`)
- [ ] Personne A : intégrer les insights dans `reports/insights.md`
- [ ] Personne B : préparer l'API pour les tests (Jour 6–7)
- [ ] Synchronisation finale post-modèle
