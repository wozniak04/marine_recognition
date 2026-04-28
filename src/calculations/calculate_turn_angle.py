import pandas as pd
import numpy as np

def calculate_turn_vectorized(lats, lons):
    """
    Oblicza kursy (bearings) i skręty (turn_angles) dla całych kolumn danych.
    Przyjmuje: lats, lons jako Series lub tablice numpy.
    Zwraca: (turn_angles, bearings) jako tablice numpy.
    """
    # Konwersja na radiany
    phi = np.radians(lats)
    lam = np.radians(lons)
    
    # Przesunięcia (shift) - obliczamy relację punktu i do punktu i+1
    phi1 = phi[:-1].values
    phi2 = phi[1:].values
    lam1 = lam[:-1].values
    lam2 = lam[1:].values
    d_lon = lam2 - lam1

    # Obliczanie Bearing (Kursu)
    y = np.sin(d_lon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lon)
    
    # Kursy w stopniach [0, 360]
    bearings = np.degrees(np.arctan2(y, x)) % 360
    
    # Aby zachować tę samą długość co wejście, dodajemy NaN na końcu (ostatni punkt nie ma następcy)
    bearings = np.append(bearings, np.nan)
    
    # Obliczanie Skrętu (Turn Angle)
    # Różnica między kursem obecnym a poprzednim
    prev_bearings = np.roll(bearings, 1) # Przesuwamy w dół
    turn = bearings - prev_bearings
    
    # Normalizacja do zakresu [-180, 180]
    turn_angles = (turn + 180) % 360 - 180
    
    # Obsługa pierwszego elementu (brak poprzedniego kursu = brak skrętu)
    turn_angles[0] = 0.0
    # Usuwamy ostatni turn_angle, bo bazował na nieistniejącym kursie
    turn_angles = np.nan_to_num(turn_angles, nan=0.0)

    return turn_angles, bearings

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