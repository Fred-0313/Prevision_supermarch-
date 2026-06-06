import pandas as pd
import numpy as np

# ==========================================================
# PARAMÈTRES
# ==========================================================
np.random.seed(42)
N_OBSERVATIONS = 5500

# ==========================================================
# MAGASINS
# ==========================================================
stores = {
    "S001": {"type": "A", "size": 4500, "city": "Cotonou", "competitor_distance": 1.2},
    "S002": {"type": "A", "size": 4200, "city": "Porto-Novo", "competitor_distance": 2.1},
    "S003": {"type": "B", "size": 3500, "city": "Parakou", "competitor_distance": 4.5},
    "S004": {"type": "B", "size": 3200, "city": "Abomey-Calavi", "competitor_distance": 1.8},
    "S005": {"type": "C", "size": 2500, "city": "Bohicon", "competitor_distance": 3.6},
    "S006": {"type": "C", "size": 2200, "city": "Lokossa", "competitor_distance": 5.1},
    "S007": {"type": "B", "size": 3400, "city": "Natitingou", "competitor_distance": 6.5},
    "S008": {"type": "A", "size": 4700, "city": "Ouidah", "competitor_distance": 2.8},
    "S009": {"type": "C", "size": 2100, "city": "Djougou", "competitor_distance": 7.3},
    "S010": {"type": "B", "size": 3300, "city": "Kandi", "competitor_distance": 8.0}
}

# ==========================================================
# DATES
# ==========================================================
dates = pd.date_range(
    start="2023-01-01",
    end="2025-12-31",
    freq="D"
)

data = []

# ==========================================================
# GÉNÉRATION
# ==========================================================
for _ in range(N_OBSERVATIONS):

    date = pd.Timestamp(np.random.choice(dates))

    store_id = np.random.choice(list(stores.keys()))
    store = stores[store_id]

    month = date.month
    day_of_week = date.dayofweek

    weekend = 1 if day_of_week >= 5 else 0
    holiday = 1 if np.random.rand() < 0.04 else 0
    school_holiday = 1 if np.random.rand() < 0.15 else 0
    promo = 1 if np.random.rand() < 0.30 else 0

    # ======================================================
    # VARIABLES ÉCONOMIQUES
    # ======================================================
    fuel_price = round(np.random.normal(680, 20), 2)
    inflation_rate = round(np.random.uniform(2.5, 8.5), 2)
    economic_index = round(np.random.uniform(90, 110), 1)

    # ======================================================
    # MÉTÉO
    # ======================================================
    if month in [12, 1, 2]:
        temperature = np.random.normal(29, 1.5)
    elif month in [3, 4, 5]:
        temperature = np.random.normal(33, 2)
    elif month in [6, 7, 8]:
        temperature = np.random.normal(27, 1.5)
    else:
        temperature = np.random.normal(30, 2)

    temperature = round(temperature, 1)

    if month in [4, 5, 6, 7, 8, 9]:
        rainfall = round(np.random.uniform(20, 120), 1)
    else:
        rainfall = round(np.random.uniform(0, 40), 1)

    # ======================================================
    # COMPOSANTES DES VENTES
    # ======================================================
    base_sales = 3500

    size_effect = store["size"] * 1.8
    promo_effect = 2500 if promo else 0
    holiday_effect = 1800 if holiday else 0
    school_effect = 800 if school_holiday else 0
    weekend_effect = 1200 if weekend else 0

    competitor_effect = store["competitor_distance"] * 180
    weather_effect = -rainfall * 6
    fuel_effect = -(fuel_price - 680) * 8

    inflation_effect = -(inflation_rate - 4) * 180
    economic_effect = (economic_index - 100) * 70

    seasonal_effect = 0
    if month in [11, 12]:
        seasonal_effect += 1800
    if month == 1:
        seasonal_effect += 1200

    noise = np.random.normal(0, 800)

    # ======================================================
    # VENTES
    # ======================================================
    sales = (
        base_sales
        + size_effect
        + promo_effect
        + holiday_effect
        + school_effect
        + weekend_effect
        + seasonal_effect
        - competitor_effect
        + weather_effect
        + fuel_effect
        + inflation_effect
        + economic_effect
        + noise
    )

    sales = max(1000, round(sales, 2))

    # ======================================================
    # STOCKAGE
    # ======================================================
    data.append([
        date,
        store_id,
        store["type"],
        store["size"],
        store["city"],
        promo,
        holiday,
        school_holiday,
        weekend,
        temperature,
        rainfall,
        fuel_price,
        inflation_rate,
        economic_index,
        store["competitor_distance"],
        sales
    ])

# ==========================================================
# DATAFRAME
# ==========================================================
columns = [
    "Date",
    "Store_ID",
    "Store_Type",
    "Store_Size",
    "City",
    "Promo",
    "Holiday",
    "School_Holiday",
    "Weekend",
    "Temperature",
    "Rainfall",
    "Fuel_Price",
    "Inflation_Rate",
    "Economic_Index",
    "Competitor_Distance",
    "Sales"
]

df = pd.DataFrame(data, columns=columns)

df = df.sort_values("Date").reset_index(drop=True)

# ==========================================================
# EXPORT
# ==========================================================
df.to_csv("supermarket_sales_forecasting.csv", index=False, encoding="utf-8")

print("Dataset généré avec succès")
print("Observations :", len(df))
print("Variables :", len(df.columns))
print(df.head())