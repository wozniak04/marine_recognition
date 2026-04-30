import pandas as pd
import torch
import numpy as np
from calculations.calculate_turn_angle import calculate_turn_vectorized
from calculations.calculate_speed import calculate_speed_haversine

def predict_maneuver(model, data_window):
    """
    data_window: tablica z kolumnami [speed, turn_angle, In Port, time_diff]
    """
    label_names = ["In Port", "At Sea Anchor", "At Sea Adrift", "At Sea Voyage"]
    model.eval()
    

    # 4. Predykcja
    with torch.no_grad():
        input_tensor = torch.tensor(data_window, dtype=torch.float32).unsqueeze(0)
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).numpy()[0]
        
        idx = np.argmax(probs)
        return label_names[idx], probs[idx] * 100