import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def create_sequences(data: np.ndarray, seq_length: int, horizon: int):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length + horizon - 1])
    return np.array(xs), np.array(ys)

def run_rf_model(df: pd.DataFrame, target_col: str, train_frac: float = 0.8, seq_length: int = 10, horizon: int = 1):
    """
    Exécute un modèle Random Forest Regressor sur la série de trafic.
    """
    series = df[target_col].values
    
    # Création des séquences (X, y)
    X, y = create_sequences(series, seq_length, horizon)
    
    if len(X) == 0:
        raise ValueError("Les données sont trop courtes pour cette séquence et cet horizon.")
    
    # Split Train/Test
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Ensemble d'entraînement ou de test vide (pas assez de données).")
    
    # Initialisation du modèle avec quelques hyperparamètres de base
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    start_time = time.time()
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédiction
    test_predictions = model.predict(X_test)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mape = mean_absolute_percentage_error(y_test, test_predictions)
    
    # Métriques métier
    under_allocation = np.mean(test_predictions < y_test) * 100
    over_allocation = np.mean(test_predictions > y_test) * 100
    
    return {
        'model_name': 'Random Forest',
        'horizon': horizon,
        'true_values': y_test,
        'predictions': test_predictions,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'under_allocation': under_allocation,
        'over_allocation': over_allocation,
        'execution_time': execution_time
    }
