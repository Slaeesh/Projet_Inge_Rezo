import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error

class MLP(nn.Module):
    def __init__(self, seq_length):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(seq_length, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def run_nn_model(df, train_frac=0.8, seq_length=10, epochs=20, batch_size=32):
    """
    Exécute un modèle de Réseau de Neurones sur la série de trafic.

    Args:
        df (pd.DataFrame): DataFrame avec la colonne 'traffic_mbps'.
        train_frac (float): Fraction de données pour l'entraînement.
        seq_length (int): Longueur de la séquence (sliding window) en entrée.
        epochs (int): Nombre d'époques d'entraînement.
        batch_size (int): Taille de batch.
        
    Returns:
        dict: Résultats, y compris MAE et temps d'exécution.
    """
    series = df['traffic_mbps'].values
    
    # Création des séquences (X, y)
    X, y = create_sequences(series, seq_length)
    
    # Split Train/Test
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Conversion en tenseurs PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # DataLoaders
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    model = MLP(seq_length)
    criterion = nn.L1Loss() # MAE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    start_time = time.time()
    
    # Mode Entraînement
    model.train()
    for __ in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    # Mode Prédiction
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    pred_np = test_predictions.squeeze().numpy()
    true_np = y_test_t.squeeze().numpy()
    
    mae = mean_absolute_error(true_np, pred_np)
    
    return {
        'model_name': 'NN (MLP PyTorch)',
        'true_values': true_np,
        'predictions': pred_np,
        'mae': mae,
        'execution_time': execution_time
    }
