import pandas as pd
import numpy as np

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return np.degrees(np.arctan2(y, x)) % 360


df = pd.read_csv('data/Ship_Operation_example_dataset_unclassified_source_GPS_1.csv').sort_values('signaldate')

# Oblicz kurs do następnego punktu
df['bearing'] = calculate_bearing(df['LAT'], df['LON'], df['LAT'].shift(-1), df['LON'].shift(-1))

# Oblicz zmianę kursu (skręt)
df['turn_angle'] = df['bearing'].shift(1) - df['bearing']

# Normalizacja kąta (-180 do 180)
df['turn_angle'] = (df['turn_angle'] + 180) % 360 - 180
print(df[['signaldate', 'LAT', 'LON', 'bearing', 'turn_angle']].head())