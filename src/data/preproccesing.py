import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from calculations.calculate_speed import calculate_speed_haversine
from calculations.calculate_turn_angle import calculate_turn_vectorized
from data.ports import check_if_in_port

def get_scaler():
    return joblib.load('scaler.pkl')

def create_windows_for_training(df, window_size=10):
    # 1. Porządki
    df['signaldate'] = pd.to_datetime(df['signaldate'])
    df = df.sort_values('signaldate').reset_index(drop=True)
    
    # Bezpieczny time_diff (jeśli 0, dajemy 60s żeby uniknąć dzielenia przez zero w prędkości)
    df['time_diff'] = df['signaldate'].diff().dt.total_seconds().fillna(60.0)
    df.loc[df['time_diff'] <= 0, 'time_diff'] = 60.0

    # 2. Prędkość i Skręt
    # Zakładamy że calculate_speed_haversine przyjmuje Series
    df['speed'] = calculate_speed_haversine(
        df['LAT'].shift(1), df['LON'].shift(1), 
        df['LAT'], df['LON'], df['time_diff']
    ).fillna(0)

    # Używamy tylko TWOJEJ funkcji - nie pozwalamy GPT nic nadpisywać
    turn_angles, bearings = calculate_turn_vectorized(df['LAT'], df['LON'])
    df['turn_angle'] = turn_angles
    df['bearing'] = bearings

    # 3. Skalowanie
    # Dodajemy time_diff do feature_cols zgodnie z Twoją prośbą
    feature_cols = ["speed", "turn_angle", "In Port", "time_diff"]
    label_cols = ["In Port", "At Sea Anchor", "At Sea Adrift", "At Sea Voyage"]
    # if df['time_diff'].nunique() <= 1:
    #     feature_cols = ["speed", "turn_angle", "In Port"]
    # Skalujemy tylko kolumny ciągłe. In Port zostawiamy w spokoju (0/1)
    continuous_features = ["speed", "turn_angle", "time_diff"]
    
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # 4. Tworzenie okien (Optymalizacja przez NumPy)
    data_feat = df[feature_cols].to_numpy(dtype=np.float32)
    data_lab = df[label_cols].to_numpy(dtype=np.float32)

    X, Y = [], []
    # Pętla po danych
    for i in range(len(df) - window_size + 1):
        window = data_feat[i : i + window_size]
        X.append(window)
        # Bierzemy label tylko z OSTATNIEGO punktu okna (Many-to-One)
        Y.append(data_lab[i + window_size - 1]) 

    return np.array(X), np.array(Y), scaler

def prepare_data_for_prediction(df):
    # 1. Porządki
    df['signaldate'] = pd.to_datetime(df['signaldate'])
    df = df.sort_values('signaldate').reset_index(drop=True)
    # Bezpieczny time_diff (jeśli 0, dajemy 60s żeby uniknąć dzielenia przez zero w prędkości)
    df['time_diff'] = df['signaldate'].diff().dt.total_seconds().fillna(60.0)
    df.loc[df['time_diff'] <= 0, 'time_diff'] = 60.0

    # 2. Prędkość i Skręt
    # Zakładamy że calculate_speed_haversine przyjmuje Series
    df['speed'] = calculate_speed_haversine(
        df['LAT'].shift(1), df['LON'].shift(1), 
        df['LAT'], df['LON'], df['time_diff']
    ).fillna(0)


    turn_angles, bearings = calculate_turn_vectorized(df['LAT'], df['LON'])
    df['turn_angle'] = turn_angles
    df['bearing'] = bearings
    df['In Port'] = df.apply(lambda row: check_if_in_port(row['LAT'], row['LON']), axis=1)
    # 3. Skalowanie

    feature_cols = ["speed", "turn_angle", "In Port", "time_diff"]
    

    continuous_features = ["speed", "turn_angle", "time_diff"]
    
    scaler = get_scaler()  # Wczytujemy scaler wytrenowany na danych treningowych
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    #data_feat = df.to_numpy(dtype=np.float32)
   

    
    return df

