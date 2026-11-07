# config.py - SEULE source de verite pour Fred et Sherif
# Ne jamais modifier un chiffre ici sans prevenir l'autre

UNIT = "milliers de FCFA (donnees simulees)"
# 1 unite = 1 000 FCFA. Prediction 10 098 = 10 098 000 FCFA

METRICS = {
    "MAE": 696,
    "RMSE": 874,
    "R2": 0.8898,
    "MAPE": 7.55,  # en %
}

# Magasins reels du dataset. Ne jamais inventer une autre ville ou surface
STORE_INFO = {
    "S001": {"Store_Type": "A", "City": "Cotonou", "Store_Size": 4700, "Competitor_Distance": 2.5},
    "S003": {"Store_Type": "B", "City": "Kandi", "Store_Size": 3300, "Competitor_Distance": 3.2},
    # a completer avec S002 a S010, avec les VRAIES valeurs du CSV
}

# Bornes pour la validation
BOUNDS = {
    "Store_Size": (2100, 4700),
    "Promo": (0, 1),
    "Holiday": (0, 1),
    "Weekend": (0, 1),
    "School_Holiday": (0, 1),
    "Temperature": (20, 38),
    "Rainfall": (0, 60),
    "Fuel_Price": (600, 750),
}
