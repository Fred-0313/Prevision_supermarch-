<<<<<<< HEAD
import pandas as pd

df = pd.read_csv("/Users/Sherif//Desktop/Prevision_supermarche/supermarket_sales_forecasting.csv")

print(df.head())

#conversion des dates
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store_ID", "Date"])
df["Date"]

=======
import prevision_supermarche as ps
import numpy as np
>>>>>>> f8dd463aa94ee458974f20e9a688eddd18cf3bed

df["year"] = df["Date"].dt.year

df["quarter"] = df["Date"].dt.quarter

df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)

df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

def saison(mois):
    if mois in [12,1,2]:
        return "Dry_Season"
    elif mois in [3,4,5]:
        return "Pre_Rainy"
    elif mois in [6,7,8,9]:
        return "Rainy"
    else:
        return "Post_Rainy"

df["season"] = df["month"].apply(saison)

df["Promo_Weekend"] = df["Promo"] * df["Weekend"]

df["Promo_Holiday"] = df["Promo"] * df["Holiday"]

df["Promo_SchoolHoliday"] = (
    df["Promo"] * df["School_Holiday"]
)

df["economic_pressure"] = (
    df["Inflation_Rate"] *
    (100/df["Economic_Index"])
)