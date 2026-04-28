import pandas as pd
import numpy as np

#SOG 
df = pd.read_csv('data/Ship_Operation_example_dataset_unclassified_source_GPS_1.csv').sort_values('signaldate')
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Promień Ziemi w metrach
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# 1. Konwersja czasu i sortowanie
df['signaldate'] = pd.to_datetime(df['signaldate'])
df = df.sort_values('signaldate')

# 2. Obliczanie różnicy czasu w sekundach
df['time_diff'] = df['signaldate'].diff().dt.total_seconds()

# 3. Obliczanie dystansu do poprzedniego punktu (w metrach)
df['distance'] = haversine_distance(
    df['LAT'].shift(1), df['LON'].shift(1),
    df['LAT'], df['LON']
)

# 4. Prędkość (V = s / t)
df['speed_m_s'] = df['distance'] / df['time_diff']  # wynik w m/s
print(df[['signaldate', 'LAT', 'LON', 'time_diff', 'distance', 'speed_m_s']].head())