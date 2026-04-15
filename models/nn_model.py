import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data: np.ndarray, seq_length: int, horizon: int):
    """
    Crée les séquences d'entraînement.
    La cible y se trouve à 'horizon' pas de temps APRÈS la fin de la séquence d'entrée.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length + horizon - 1])
    return np.array(xs), np.array(ys)

def run_nn_model(df: pd.DataFrame, target_col: str, train_frac: float = 0.8, seq_length: int = 10, horizon: int = 1, epochs: int = 20, batch_size: int = 32, model_type: str = 'gru'):
    """
    Exécute un modèle de Réseau de Neurones sur la série de trafic.

    Args:
        df (pd.DataFrame): DataFrame global.
        target_col (str): Nom de la colonne faisceau ciblé.
        train_frac (float): Fraction de données pour l'entraînement.
        seq_length (int): Longueur de la séquence.
        horizon (int): Horizon de prédiction k.
        epochs (int): Nombre d'époques d'entraînement.
        batch_size (int): Taille de batch.
        model_type (str): Type de modèle ('gru' ou 'lstm').
        
    Returns:
        dict: contenant les métriques complètes.
    """
    series = df[target_col].values
    
    # Création des séquences (X, y) en tenant compte de l'horizon
    X, y = create_sequences(series, seq_length, horizon)
    
    if len(X) == 0:
        raise ValueError("Les données sont trop courtes pour cette séquence et cet horizon.")
    
    # Split Train/Test
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Ensemble d'entraînement ou de test vide.")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    if model_type == 'gru':
        model = GRUModel()
        model_name = 'GRU PyTorch'
    elif model_type == 'lstm':
        model = LSTMModel()
        model_name = 'LSTM PyTorch'
    else:
        raise ValueError("model_type doit être 'gru' ou 'lstm'")
        
    criterion = nn.L1Loss() # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    start_time = time.time()
    
    # Entraînement
    model.train()
    for __ in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    # Prédiction
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    pred_np = test_predictions.squeeze(-1).numpy()
    true_np = y_test_t.squeeze(-1).numpy()
    
    mae = mean_absolute_error(true_np, pred_np)
    
    # Métriques métier
    under_allocation = np.mean(pred_np < true_np) * 100
    over_allocation = np.mean(pred_np > true_np) * 100
    
    return {
        'model_name': model_name,
        'horizon': horizon,
        'true_values': true_np,
        'predictions': pred_np,
        'mae': mae,
        'under_allocation': under_allocation,
        'over_allocation': over_allocation,
        'execution_time': execution_time
    }
