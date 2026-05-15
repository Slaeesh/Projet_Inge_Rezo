import time
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def create_sequences(data: np.ndarray, seq_length: int, horizon: int):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length + horizon - 1])
    return np.array(xs), np.array(ys)

def run_svm_model(df: pd.DataFrame, target_col: str, train_frac: float = 0.8, seq_length: int = 10, horizon: int = 1):
    """
    Exécute un modèle SVM (Support Vector Regressor) sur la série de trafic.
    """
    series = df[target_col].values
    
    # 3. Sécurisation de la taille du Dataset vs Horizon
    if len(series) < seq_length + horizon + 10:
        print(f"⚠️ AVERTISSEMENT (SVM) : Le dataset est trop petit ({len(series)} lignes) pour la séquence ({seq_length}) et l'horizon ({horizon}).")
        print("L'horizon est automatiquement réduit à 1.")
        horizon = 1
        seq_length = min(seq_length, max(1, len(series) // 3))
        if len(series) < seq_length + horizon + 2:
            raise ValueError(f"Dataset ridiculement petit ({len(series)}). Impossible de continuer.")
            
    # Création des séquences (X, y)
    X, y = create_sequences(series, seq_length, horizon)
    
    # Split Train/Test
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Ensemble d'entraînement ou de test vide (pas assez de données).")
        
    # 1. Normalisation obligatoire
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_test_scaled = scaler_X.transform(X_test)
    
    # Initialisation du modèle
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    start_time = time.time()
    
    # Entraînement
    model.fit(X_train_scaled, y_train_scaled)
    
    # Prédiction
    test_predictions_scaled = model.predict(X_test_scaled)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 1. (Suite) Inverse transform pour retrouver les Mbps
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_test, test_predictions)
    
    # Métriques métier
    under_allocation = np.mean(test_predictions < y_test) * 100
    over_allocation = np.mean(test_predictions > y_test) * 100
    
    return {
        'model_name': 'SVM (Support Vector Regressor)',
        'horizon': horizon,
        'true_values': y_test,
        'predictions': test_predictions,
        'mae': mae,
        'under_allocation': under_allocation,
        'over_allocation': over_allocation,
        'execution_time': execution_time
    }
