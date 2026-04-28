import pandas as pd
import numpy as np

#SOG 
#df = pd.read_csv('training_data/Ship_Operation_example_dataset_unclassified_source_GPS_1.csv').sort_values('signaldate')

def calculate_speed_haversine(lat1, lon1, lat2, lon2, time_interval_sec=60):
    """
    Oblicza prędkość (m/s) między dwoma punktami na podstawie wzoru Haversine.
    Domyślny interwał czasowy to 60 sekund.
    """
    R = 6371000  # Promień Ziemi w metrach
    
    # Konwersja na radiany
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    # Wzór Haversine (obliczanie dystansu 'c')
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    
    # Obliczanie prędkości: V = s / t
    speed_m_s = distance / time_interval_sec
    
    return speed_m_s

# przykładowe użycie:
# df['signaldate'] = pd.to_datetime(df['signaldate'])
# df = df.sort_values('signaldate')



# print (df[0:2].head())
# print(calculate_speed_haversine(df['LAT'].iloc[0], df['LON'].iloc[0], df['LAT'].iloc[1], df['LON'].iloc[1]))