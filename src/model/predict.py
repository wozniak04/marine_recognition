import pandas as pd
import torch
import numpy as np
from calculations.calculate_turn_angle import calculate_turn_vectorized
from calculations.calculate_speed import calculate_speed_haversine

def predict_maneuver(model, gps_window, scaler, label_names):
    """
    gps_window: DataFrame lub tablica z kolumnami [LAT, LON, In Port, (opcjonalnie signaldate)]
    """
    model.eval()
    
    # 1. Obliczamy parametry fizyczne (tak jak w create_windows)
    # Zakładamy, że gps_window to DataFrame z 10 wierszami
    df_temp = gps_window.copy()
    
    # Obliczamy time_diff (jeśli nie masz daty, zakładamy 60s)
    if 'signaldate' in df_temp.columns:
        df_temp['signaldate'] = pd.to_datetime(df_temp['signaldate'])
        df_temp['time_diff'] = df_temp['signaldate'].diff().dt.total_seconds().fillna(60.0)
    else:
        df_temp['time_diff'] = 60.0

    # Prędkość i Skręt (Używamy Twoich funkcji!)
    df_temp['speed'] = calculate_speed_haversine(
        df_temp['LAT'].shift(1), df_temp['LON'].shift(1), 
        df_temp['LAT'], df_temp['LON'], df_temp['time_diff']
    ).fillna(0)
    
    turn_angles, _ = calculate_turn_vectorized(df_temp['LAT'], df_temp['LON'])
    df_temp['turn_angle'] = turn_angles

    # 2. Selekcja kolumn i SKALOWANIE
    # Musi być ta sama kolejność co w treningu!
    continuous_features = ["speed", "turn_angle", "time_diff"]
    
    # Skalujemy kolumny ciągłe
    scaled_values = scaler.transform(df_temp[continuous_features])
    
    # 3. Składanie finalnego wektora (np. speed, turn, in_port, time_diff)
    # Uwaga: In Port pobieramy z surowych danych (nie skalujemy go)
    in_port = df_temp['In Port'].to_numpy().reshape(-1, 1)
    
    # Składamy zgodnie z Twoim feature_cols: [speed, turn, port, time]
    final_input = np.hstack([
        scaled_values[:, 0:1], # speed
        scaled_values[:, 1:2], # turn_angle
        in_port,               # In Port
        scaled_values[:, 2:3]  # time_diff
    ])

    # 4. Predykcja
    with torch.no_grad():
        input_tensor = torch.tensor(final_input, dtype=torch.float32).unsqueeze(0)
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).numpy()[0]
        
        idx = np.argmax(probs)
        return label_names[idx], probs[idx] * 100