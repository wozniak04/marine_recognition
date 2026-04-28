import torch
import numpy as np
def predict_maneuver(model, window_data, scaler, label_names):
    """
    window_data: surowe okno (np. 10 wierszy z [speed, turn_angle, In Port])
    scaler: dopasowany wcześniej StandardScaler
    label_names: lista nazw klas (np. ["Outlier", "Port", "Anchor", "Adrift", "Voyage"])
    """
    model.eval()
    with torch.no_grad():
        # 1. Skalowanie surowych danych (tylko kolumny ciągłe: speed i turn)
        # window_data powinna mieć kształt (window_size, features)
        # Zakładamy, że In Port to ostatnia kolumna i jej nie skalujemy
        continuous = window_data[:, [0, 1, 3]] 
        scaled_continuous = scaler.transform(continuous)
        
        # Łączymy z powrotem z flagą In Port
        in_port_col = window_data[:, 2].reshape(-1, 1)
        final_input = np.hstack([scaled_continuous, in_port_col])
        
        # 2. Konwersja na tensor i dodanie wymiaru batch (1, seq, feat)
        input_tensor = torch.tensor(final_input, dtype=torch.float32).unsqueeze(0)
        
        # 3. Predykcja
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).numpy()[0] # Prawdopodobieństwo dla każdej klasy
        
        # 4. Wynik
        best_class_idx = np.argmax(probabilities)
        confidence = probabilities[best_class_idx] * 100
        
        return label_names[best_class_idx], confidence