import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, Y_train, epochs=50, batch_size=32, lr=0.001):
    # Konwersja na tensory PyTorch
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss() # Dobry wybór dla klasyfikacji wieloetykietowej
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

    # Zapisywanie modelu
    torch.save(model.state_dict(), "ship_model.pth")
    print("Model zapisany jako ship_model.pth")