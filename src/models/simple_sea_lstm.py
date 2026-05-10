import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleSeaLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # Dodajemy bidirectional=True - sieć czyta sekwencję z przeszłości w przyszłość i na odwrót
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)
        
        # Ponieważ mamy 2 kierunki (przód i tył), warstwa liniowa musi przyjąć 2 razy więcej ukrytych cech
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Mean Pooling: bierzemy średnią ze wszystkich stanów w oknie (zamiast tylko ostatniego).
        # Daje to sieci pełen kontekst i ignoruje punktowe uderzenia wiatru na kotwicy.
        out_mean = torch.mean(out, dim=1)
        return self.fc(out_mean)

class SimpleSeaModelWrapper:
    def __init__(self, epochs=20, lr=0.001, class_weights=None):
        self.epochs = epochs
        self.lr = lr
        self.model = SimpleSeaLSTM()
        
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Dodajemy weight_decay (Regularyzacja L2), by zapobiec tzw. overfittingowi do szumów GPS
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # Dodajemy scheduler zmniejszający Learning Rate, gdy strata przestaje spadać
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        if len(X_seq) == 0: return
        
        self.model.train()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        # Większy batch_size to stabilniejsze gradienty, zwłaszcza przy małych klasach
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_x), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
            # Krok schedulera (jeśli strata uśredniona dla epoki przestaje maleć, zwolnij)
            self.scheduler.step(epoch_loss / len(loader))

    def predict(self, X_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(X_seq) == 0: return np.array([]), np.array([])
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            logits = self.model(X_tensor)
            # Używamy softmax, aby uzyskać prawdopodobieństwa
            probs = torch.softmax(logits, dim=1)
            # Wybieramy pewność (najwyższe prawdopodobieństwo) i indeks klasy
            confs, preds = torch.max(probs, dim=1)
            
        return preds.numpy(), confs.numpy() * 100.0