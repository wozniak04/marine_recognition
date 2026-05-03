import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler  
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import ModelDataset
from src.utils.logger import create_logger


LEARNING_RATE   = 0.01
EPOCHS          = 1000
BATCH_SIZE      = 64
TEST_SIZE       = 0.3
RANDOM_STATE    = 128


logger = create_logger("MLP")

def build_model(input_size: int, hidden_sizes: list[int], output_size: int) -> nn.Module:
    """
    Builds configurable MLP.
    Args:
      - input_size: features amount
      - hidden_sizes: hidden layers' architecture e.g. [16, 32, 8]
      - output_size: labels amount
    """
    layers: list[nn.Module] = []
    prev = input_size
    for h in hidden_sizes:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, output_size))

    # Define model
    class _NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(*layers)

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
    return _NeuralNetwork()

def _train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> tuple[float, float]:
    model.train()
    train_loss, train_acc = 0.0, 0.0
    
    batches = len(dataloader)
    size = len(dataloader.dataset)
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    return train_loss / batches, train_acc / size

def _validate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    total_loss, correct = 0.0, 0
    predictions: list[int] = []
    true_values: list[int] = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            predictions.extend(pred.argmax(1).cpu().numpy())
            true_values.extend(y.cpu().numpy())
    val_loss = total_loss / num_batches
    val_acc = correct / size

    return val_loss, val_acc, predictions, true_values



class TrainResult(dict):
    model:              nn.Module
    label_encoder:      LabelEncoder
    scaler:             StandardScaler
    accuracy:           float
    confusion_matrix:   pd.DataFrame



# Main
def train_classifier(
    df: pd.DataFrame,
    hidden_sizes: list[int],
    feature_names: list[str],
    class_names: list[str] | None = None,
    target_col: str | None = None,
    target_classes: list[str | int] | None = None,
    plot_path: str | None = "visualization/training_plot.png"
) -> TrainResult:
    """
    Trains MLP on any DataFrame.
    Args:
      - df:             DataFrame with features and labels
      - hidden_sizes:   hidden layers' sizes e.g. [16, 32, 8]
      - feature_names:  columns to use as input features
      - class_names:    human-readable class labels (None = taken from LabelEncoder)
      - target_col:     target column name (None = last column)
      - target_classes: specify classes used for training
      - plot_path:      path to save the training plot (None = skip saving)
    Returns TrainingResult with keys: model, label_encoder,
        scaler, accuracy, confusion_matrix.
    """
    logger.info(f"Starting MLP training.")
    logger.info(f"Config: epochs={EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE}, test_size={TEST_SIZE}")

    if target_col is None:
        target_col = df.columns[-1]
        
    if "is_outlier" in df.columns:
        before = len(df)
        df = df[df["is_outlier"] != 1].reset_index(drop=True)
        logger.info(f"Removed {before - len(df)} outliers ({len(df)} samples remaining).")
        
    if target_classes is not None:
        before = len(df)
        df = df[df[target_col].isin(target_classes)].reset_index(drop=True)
        logger.info(f"Filtering classes {target_classes}: {before - len(df)} rows removed ({len(df)} remaining).")
    
    X: np.ndarray = df[feature_names].values
    y_raw: np.ndarray = df[target_col].values
    
    le = LabelEncoder()
    y: np.ndarray = le.fit_transform(y_raw)
    num_classes: int = len(le.classes_)
    input_size: int = X.shape[1]
    
    logger.info(f"Dataset: {len(df)} samples | {input_size} features | {num_classes} classes: {list(le.classes_)}")
    
    if class_names is None:
        class_names = list(le.classes_)
        
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_val)} val samples")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    train_loader = DataLoader(ModelDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ModelDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device.")

    model = build_model(input_size, hidden_sizes, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"Model architecture:\n{model}")

    train_losses: list[float] = []
    val_losses:   list[float] = []
    train_accs:   list[float] = []
    val_accs:     list[float] = []

    predictions: list[int] = []
    true_values: list[int] = []

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = _train(train_loader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
    
        val_loss, val_acc, predictions, true_values = _validate(val_loader, model, loss_fn, device)
        val_losses.append(val_loss)
        val_accs.append(val_loss)
        
        if(epoch)%10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>3}/{EPOCHS}  |  "
                f"Train => loss: {train_loss:.4f}, acc: {train_acc:.4f}  |  "
                f"Val => loss: {val_loss:.4f}, acc: {val_acc:.4f}"
            )

    logger.info(f"Training and validation finished.")
    
    if plot_path is not None:
        epochs_range = range(1, EPOCHS + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs_range, train_losses, label="Train")
        ax1.plot(epochs_range, val_losses,   label="Val")
        ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
        ax1.legend(); ax1.grid(True)

        ax2.plot(epochs_range, train_accs, label="Train")
        ax2.plot(epochs_range, val_accs,   label="Val")
        ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Training plot saved to: {plot_path}")


    acc = accuracy_score(y_val, predictions)
    conf_mat = confusion_matrix(y_val, predictions)
    conf_df = pd.DataFrame(conf_mat, index=class_names, columns=class_names)

    logger.info(
        f"Experiment results:\n"
        f"  Accuracy: {round(acc * 100, 4)}%\n"
        f"  Confusion matrix:\n{conf_df}"
    )
    
    return TrainResult(
        model=model,
        label_encoder=le,
        scaler=scaler,
        accuracy=acc,
        confusion_matrix=conf_df,
    )