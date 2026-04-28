import pandas as pd
df = pd.read_csv('data/processed/Ship_operation.csv')

# print(df.iloc[(df["acc"] > 0.05) | (df["acc"] < -0.05)])
# outliers = df[(df["acc"] > 0.03) | (df["acc"] < -0.03)]
# print(len(outliers))
# print(df["acc"].quantile([0.01, 0.05, 0.95, 0.99]))


col = "velocity_m_s"

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

dolna_granica = Q1 - 1.5 * IQR
gorna_granica = Q3 + 1.5 * IQR
print(dolna_granica)
print(gorna_granica)

df = df[(df["velocity_m_s"] >= dolna_granica) & (df["velocity_m_s"] <= gorna_granica)]
print(df.describe())

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns


# # print(df.describe())
# for i in ["velocity_m_s", "dvelocity_m_s", "dCOG", "acc", "velocity_knot"]:
#     print(f"Dla {i}")
#     for p in [70, 75, 80, 82, 84, 86, 88, 90, 95, 99, 99.5, 99.6, 99.7, 99.8, 99.9]:
#         alpha = (100 - p) / 200
#         # nanquantile ignoruje braki danych
#         low, high = np.nanquantile(df[i], [alpha, 1 - alpha])
#         print(f"  {p}% - {low}, {high}")
        
