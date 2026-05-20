import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

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
    """
    series = df[target_col].values
    
    # 3. Sécurisation de la taille du Dataset vs Horizon
    if len(series) < seq_length + horizon + 10:
        print(f"⚠️ AVERTISSEMENT (DL) : Le dataset est trop petit ({len(series)} lignes) pour la séquence ({seq_length}) et l'horizon ({horizon}).")
        print("L'horizon est automatiquement réduit à 1 pour permettre l'entraînement.")
        horizon = 1
        seq_length = min(seq_length, max(1, len(series) // 3))
        if len(series) < seq_length + horizon + 2:
            raise ValueError(f"Dataset ridiculement petit ({len(series)}). Impossible de continuer.")
            
    # Création des séquences (X, y) en tenant compte de l'horizon
    X, y = create_sequences(series, seq_length, horizon)
    
    # Split Train/Test (AVANT LA NORMALISATION pour éviter le data leakage)
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Ensemble d'entraînement ou de test vide après le split.")
        
    # 1. Normalisation obligatoire (MinMaxScaler sur Train, puis appliqué sur Test)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_test_scaled = scaler_X.transform(X_test)
    # On ne normalise pas y_test ici, on le garde en Mbps pour calculer la vraie erreur à la fin.
    
    # 4. Ajustement pour les petits datasets (Deep Learning)
    lr = 0.01
    if len(X_train) < 200:
        epochs = max(epochs, 150) # On force plus d'époques si le dataset est tout petit
        lr = 0.005 # Learning rate ajusté
        
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
            
    # Prédiction sur le Test set (valeurs normalisées)
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_t).squeeze(-1).numpy()
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 1. (Suite) Inverse Transform pour récupérer les Mbps (Série réelle)
    pred_np = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
    true_np = y_test # Vérité terrain déjà en Mbps
    
    mae = mean_absolute_error(true_np, pred_np)
    rmse = np.sqrt(mean_squared_error(true_np, pred_np))
    mape = mean_absolute_percentage_error(true_np, pred_np)
    
    # Métriques métier
    under_allocation = np.mean(pred_np < true_np) * 100
    over_allocation = np.mean(pred_np > true_np) * 100
    
    return {
        'model_name': model_name,
        'horizon': horizon,
        'true_values': true_np,
        'predictions': pred_np,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'under_allocation': under_allocation,
        'over_allocation': over_allocation,
        'execution_time': execution_time
    }
