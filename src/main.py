import joblib
import torch
from data.load_data import load_data
from data.preproccesing import create_windows_for_training, prepare_data_for_prediction
from model.example_neural_network import ShipManeuverLSTM
from model.training import train_model
from model.predict import predict_maneuver
import pandas as pd
import numpy as np

def calculate_accuracy_report(df):
    # 1. Lista Twoich klas
    classes = ["In Port", "At Sea Anchor", "At Sea Adrift", "At Sea Voyage"]
    
    # 2. Wyciągamy prawdziwy stan (idxmax zwraca nazwę kolumny z 1)
    df['true_state'] = df[classes].idxmax(axis=1)
    
    # 3. Porównanie - poprawione użycie .str.strip()
    # Dodatkowo rzutujemy na stringi, by uniknąć problemów z typami
    pred = df['predicted_state'].astype(str).str.strip()
    true = df['true_state'].astype(str).str.strip()
    df['is_correct'] = pred == true
    
    # 4. Liczenie statystyk
    total_samples = len(df)
    correct_samples = df['is_correct'].sum()
    overall_accuracy = (correct_samples / total_samples) * 100 if total_samples > 0 else 0
    
    # 5. Raport klasowy (zapewniamy, że wszystkie 4 klasy się wyświetlą)
    class_reports = []
    for cls in classes:
        subset = df[df['true_state'] == cls]
        count = len(subset)
        correct = subset['is_correct'].sum() if count > 0 else 0
        acc = (correct / count * 100) if count > 0 else 0
        
        class_reports.append({
            'Klasa': cls,
            'Próbek': count,
            'Poprawnie': correct,
            'Skuteczność': f"{acc:.2f}%"
        })
    
    # Wyświetlanie raportu
    print("-" * 60)
    print(f"OGÓLNA SKUTECZNOŚĆ MODELU: {overall_accuracy:.2f}%")
    print("-" * 60)
    print(pd.DataFrame(class_reports).to_string(index=False))
    print("-" * 60)

    # 6. Wybór kolumn do zwrócenia
    selected_columns = [
        'signaldate', 'LAT', 'LON', 
        'In Port', 'At Sea Anchor', 'At Sea Adrift', 'At Sea Voyage', 
        'predicted_state', 'confidence', 'true_state', 'is_correct'
    ]
    
    return df[selected_columns].copy()

# Użycie:
# df = calculate_accuracy_report(df)

def makePredictionForAllDataset(model,data):
    model.eval()
    # Przygotuj dane do predykcji
    data_feat = prepare_data_for_prediction(data)
    # Przechowuj wyniki
    predictions = []

    # Przewiń przez dane okno po oknie
    for i in range(len(data_feat) - 10 + 1):
        window = data_feat[i : i + 10] # Okno o rozmiarze 10
        state, conf = predict_maneuver(model, window[["speed", "turn_angle", "In Port", "time_diff"]].to_numpy(dtype=np.float32))
        predictions.append((state, conf))
        data_feat.at[i, 'predicted_state'] = state
        data_feat.at[i, 'confidence'] = conf
    
    return data_feat


def ltrain_model(model,df, epochs=100):
    X, Y, my_scaler = create_windows_for_training(df, window_size=10)
    model.train()
    train_model(model, X, Y, epochs=epochs)
    joblib.dump(my_scaler, 'scaler.pkl')  

df=load_data('training_data/Ship_Operation_example_dataset_classified_2.csv')
df2=load_data('training_data/Ship_Operation_example_dataset_classified.csv')
df3=load_data('training_data/filtered_ship_data.csv')
df4=load_data('training_data/filtered_ship_data_2.csv')
model = ShipManeuverLSTM(input_size=4, hidden_size=64, num_classes=4)
ltrain_model(model,pd.concat([df2, df3, df4]), epochs=100)
#model.load_state_dict(torch.load("ship_model.pth", weights_only=True))
predictions=pd.DataFrame(makePredictionForAllDataset(model,df))
calculate_accuracy_report(predictions).to_csv("predictions.csv", index=False)

# 3. Przykład predykcji na nowym oknie
# labels = ["Outlier GPS", "In Port", "At Sea Anchor", "At Sea Adrift", "At Sea Voyage"]
# sample_window = df2.iloc[9600:9600+10]
# (W praktyce w predict podawałbyś surowe dane z GPS)
