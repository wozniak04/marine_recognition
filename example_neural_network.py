import torch
import torch.nn as nn

class ShipManeuverLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(ShipManeuverLSTM, self).__init__()
        
        # Inicjalizacja LSTM
        # batch_first=True oznacza, że podajemy dane w formacie (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 # Zapobiega przeuczeniu
        )
        
        # W pełni połączona warstwa klasyfikująca
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_size)
        
        # Przejście przez LSTM
        # out zawiera stany ukryte dla KAŻDEGO kroku czasowego
        out, (hidden, cell) = self.lstm(x)
        
        # Interesuje nas tylko wniosek po przeanalizowaniu CAŁEJ sekwencji,
        # więc bierzemy wyjście z ostatniego kroku czasowego (index -1)
        last_out = out[:, -1, :] 
        
        # Przepuszczenie przez warstwę klasyfikacyjną
        # Zwraca wektor prawdopodobieństw (tzw. logity) dla każdej klasy manewru
        final_prediction = self.fc(last_out)
        
        return final_prediction

# Przykładowa inicjalizacja modelu:
# - 4 cechy (SOG, DeltaCOG, is_in_port, delta_time)
# - 64 neurony w warstwie ukrytej
# - 8 klas na wyjściu (Voyage, Anchor, Adrift, Shifting, Turn, Departure, Arrival, In_Port)
model = ShipManeuverLSTM(input_size=4, hidden_size=64, num_classes=8)