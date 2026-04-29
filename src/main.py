import torch
from data.load_data import load_data
from data.preproccesing import create_windows
from model.example_neural_network import ShipManeuverLSTM
from model.training import train_model
from model.predict import predict_maneuver

df=load_data('training_data/Ship_Operation_example_dataset_classified_2.csv')
df2=load_data('training_data/Ship_Operation_example_dataset_classified.csv')
X, Y, my_scaler = create_windows(df, window_size=10)

# 2. Zainicjalizuj i wytrenuj model
# input_size=4 (speed, turn, in_port, time_diff), num_classes=5
model = ShipManeuverLSTM(input_size=4, hidden_size=64, num_classes=5)
#train_model(model, X, Y, epochs=100)
model.load_state_dict(torch.load("ship_model.pth", weights_only=True))
model.eval()
# 3. Przykład predykcji na nowym oknie
labels = ["Outlier GPS", "In Port", "At Sea Anchor", "At Sea Adrift", "At Sea Voyage"]
sample_window = df2.iloc[13624:13624+10]
# (W praktyce w predict podawałbyś surowe dane z GPS)
print("prawda: ",Y[14],flush=True) # Prawdziwe etykiety dla tego okna
state, conf = predict_maneuver(model, sample_window, my_scaler, labels)
print(f"Stan: {state} ({conf:.2f}%)")