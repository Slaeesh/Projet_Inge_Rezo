import time
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def run_arima_model(df: pd.DataFrame, target_col: str, train_frac: float = 0.8, order: tuple = (5, 1, 0), horizon: int = 1):
    """
    Exécute un modèle ARIMA sur la série de trafic.

    Args:
        df (pd.DataFrame): DataFrame global.
        target_col (str): Colonne faisceau ciblé.
        train_frac (float): Fraction de données pour l'entraînement.
        order (tuple): Paramètres (p, d, q).
        horizon (int): Horizon de prédiction k.
    """
    warnings.filterwarnings("ignore")
    series = df[target_col].values
    
    split_idx = int(len(series) * train_frac)
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    
    if len(train_data) <= max(order[0], order[2]):
        raise ValueError("Erreur : la taille des données d'entraînement est trop petite pour l'ordre ARIMA demandé.")
    
    start_time = time.time()
    
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    start_pred = len(train_data)
    end_pred = len(train_data) + len(test_data) - 1
    
    raw_preds = model_fit.predict(start=start_pred, end=end_pred, dynamic=False)
    
    if len(raw_preds) > horizon - 1:
        predictions = raw_preds[:len(raw_preds) - horizon + 1]
        true_values = test_data[horizon - 1 :]
    else:
        predictions = np.array([])
        true_values = np.array([])
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    if len(predictions) > 0:
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mape = mean_absolute_percentage_error(true_values, predictions)
        under_allocation = np.mean(predictions < true_values) * 100
        over_allocation = np.mean(predictions > true_values) * 100
    else:
        mae, rmse, mape, under_allocation, over_allocation = 0.0, 0.0, 0.0, 0.0, 0.0
    
    return {
        'model_name': f'ARIMA{order}',
        'horizon': horizon,
        'true_values': true_values,
        'predictions': predictions,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'under_allocation': under_allocation,
        'over_allocation': over_allocation,
        'execution_time': execution_time
    }
