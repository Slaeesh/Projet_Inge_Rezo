import time
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error
import pandas as pd
import warnings

def run_ar_model(df, train_frac=0.8, lags=10):
    """
    Exécute un modèle AutoRegressif (AR) sur la série de trafic.

    Args:
        df (pd.DataFrame): DataFrame indexé par temps, avec colonne 'traffic_mbps'.
        train_frac (float): Fraction de données pour l'entraînement (le reste pour test).
        lags (int): Nombre de retards (lags) pour le modèle AR.
        
    Returns:
        dict: contenant les prédictions, les vraies valeurs de test, MAE, et temps d'exécution.
    """
    # Ignorer les avertissements de statsmodels concernant l'absence de fréquence explicite dans certains index
    warnings.filterwarnings("ignore")
    
    series = df['traffic_mbps'].values
    
    # Split Train/Test
    split_idx = int(len(series) * train_frac)
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    
    if len(train_data) <= lags:
        raise ValueError("Erreur : la taille des données d'entraînement est trop petite par rapport aux lags demandés.")
    
    start_time = time.time()
    
    # Entraînement
    model = AutoReg(train_data, lags=lags)
    model_fit = model.fit()
    
    # Prédiction pour l'ensemble de test
    # On commence la prédiction juste après train_data (index split_idx)
    # Et on s'arrête à la fin de test_data
    start_pred = len(train_data)
    end_pred = len(train_data) + len(test_data) - 1
    
    predictions = model_fit.predict(start=start_pred, end=end_pred, dynamic=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Évaluation de la précision (Erreur Absolue Moyenne - MAE)
    mae = mean_absolute_error(test_data, predictions)
    
    return {
        'model_name': f'AR(lags={lags})',
        'true_values': test_data,
        'predictions': predictions,
        'mae': mae,
        'execution_time': execution_time
    }
