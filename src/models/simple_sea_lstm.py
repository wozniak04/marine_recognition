import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleSeaLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_classes=3):
        super().__init__()
        # Prosta, jednokierunkowa sieć LSTM (o wiele mniej kodu)
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        # Bierzemy wynik tylko z ostatniego punktu okna czasowego
        return self.fc(out[:, -1, :])

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
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        if len(X_seq) == 0: return
        
        self.model.train()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_x), batch_y)
                loss.backward()
                self.optimizer.step()

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