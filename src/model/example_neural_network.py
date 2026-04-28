import torch
import torch.nn as nn
import torch.nn.functional as F

class ShipManeuverLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(ShipManeuverLSTM, self).__init__()
        
        # Normalizacja wejścia - pomaga, gdy prędkość jest w m/s a kąty w stopniach
        self.layer_norm = nn.LayerNorm(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Klasyfikator
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, return_confidence=False):
        # x: (batch, seq_len, features)
        x = self.layer_norm(x)
        out, _ = self.lstm(x)
        
        # Wybieramy tylko ostatni krok czasowy (Many-to-One)
        # To sprawi, że wyjście będzie miało wymiar [batch, num_classes]
        last_step = out[:, -1, :] 
        logits = self.fc(last_step) 
        
        if return_confidence:
            # Używamy softmax, jeśli klasy się wykluczają, 
            # lub sigmoid, jeśli statek może robić kilka rzeczy naraz
            confidence = torch.sigmoid(logits) 
            return logits, confidence
            
        return logits

# Inicjalizacja pod Twoje dane:
# input_size=4 (np. SOG, turn_angle, is_in_port, time_diff)
model = ShipManeuverLSTM(input_size=4, hidden_size=128, num_classes=8, num_layers=2)