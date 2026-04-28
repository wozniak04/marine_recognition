import pandas as pd
import numpy as np

def calculate_turn(lat1, lon1, lat2, lon2, previous_bearing=None):
    """
    Oblicza skręt. Jeśli previous_bearing jest None, zwraca 0 jako skręt 
    oraz nowo obliczony kurs.
    """
    # 1. Obliczamy aktualny kurs na podstawie ruchu (zawsze potrzebujemy 2 punktów)
    phi1, lam1 = np.radians(lat1), np.radians(lon1)
    phi2, lam2 = np.radians(lat2), np.radians(lon2)
    
    d_lon = lam2 - lam1
    y = np.sin(d_lon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lon)
    
    current_bearing = np.degrees(np.arctan2(y, x)) % 360
    
    # 2. Obsługa przypadku "startu" (brak poprzedniego kursu)
    if previous_bearing is None or np.isnan(previous_bearing):
        return 0.0, current_bearing  # Brak skrętu na starcie
    
    # 3. Obliczamy skręt jeśli mamy dane
    turn = current_bearing - previous_bearing
    turn_angle = (turn + 180) % 360 - 180
    
    return turn_angle, current_bearing

# Przykładowe użycie:
# df = pd.read_csv('training_data/Ship_Operation_example_dataset_unclassified_source_GPS_1.csv').sort_values('signaldate')
# df['signaldate'] = pd.to_datetime(df['signaldate'])
# df = df.sort_values('signaldate')
# turn_angles = []
# previous_bearing = None
# for i in range(5):
#     lat1, lon1 = df['LAT'].iloc[i], df['LON'].iloc[i]
#     lat2, lon2 = df['LAT'].iloc[i+1], df['LON'].iloc[i+1]
#     print(f"Point {i} to {i+1}: ({lat1}, {lon1}) -> ({lat2}, {lon2})")
#     turn_angle, previous_bearing = calculate_turn(lat1, lon1, lat2, lon2, previous_bearing)
#     turn_angles.append(turn_angle)
# print(turn_angles)