import os
import sys
import numpy as np
import pandas as pd
import warnings

# Configuration du chemin d'accès
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import load_real_traffic_data
from data_aggregator import aggregate_data
from models.ar_model import run_ar_model
from models.arima_model import run_arima_model
from models.rf_model import run_rf_model
from models.svm_model import run_svm_model
from models.nn_model import run_nn_model
from plot_utils import plot_single_result, plot_comparison, plot_cross_link, plot_cross_beam, plot_monte_carlo

def compute_additional_metrics(res):
    if not res:
        return res
    true = np.array(res['true_values']).flatten()
    pred = np.array(res['predictions']).flatten()
    if len(true) == 0 or len(pred) == 0:
        res['rmse'] = 0.0
        res['mape'] = 0.0
        res['r2'] = 0.0
        return res
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mape = np.mean(np.abs((true - pred) / np.clip(true, 1e-5, None)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    res['rmse'] = rmse
    res['mape'] = mape
    res['r2'] = r2
    return res

def main():
    warnings.filterwarnings("ignore")
    print("=== Régénération des graphiques avec paramètres harmonisés ===")
    
    # 1. Chargement des données Scenario 1, Forward Link et Return Link
    file_path_forward = "../PRED_TRAFFIC/nb_variable_utilisateurs/scenario1/tx_throughput.csv"
    file_path_return = "../PRED_TRAFFIC/nb_variable_utilisateurs/scenario1/rx_throughput.csv"
    
    df_forward = load_real_traffic_data("forward", file_path_forward)
    df_return = load_real_traffic_data("return", file_path_return)
    
    # 2. Agrégation à 1 minute (pour harmoniser la granularité temporelle)
    freq = "1min"
    df_forward_agg = aggregate_data(df_forward, freq)
    df_return_agg = aggregate_data(df_return, freq)
    
    target_beam = "beam_1"
    seq_len = 10
    horizon = 1
    
    # 3. Génération des graphiques Mode 1 (Analyse Classique)
    print("\n--- Génération des graphiques individuels (Mode 1) ---")
    
    # ARIMA(5, 1, 0)
    print("Modèle ARIMA...")
    res_arima = run_arima_model(df_forward_agg, target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon)
    res_arima = compute_additional_metrics(res_arima)
    plot_single_result(res_arima)
    
    # Random Forest
    print("Modèle Random Forest...")
    res_rf = run_rf_model(df_forward_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
    res_rf = compute_additional_metrics(res_rf)
    plot_single_result(res_rf)
    
    # SVR (SVM)
    print("Modèle SVM/SVR...")
    res_svr = run_svm_model(df_forward_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
    res_svr = compute_additional_metrics(res_svr)
    plot_single_result(res_svr)
    
    # LSTM PyTorch
    print("Modèle LSTM PyTorch...")
    res_lstm = run_nn_model(df_forward_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=25, model_type='lstm')
    res_lstm = compute_additional_metrics(res_lstm)
    plot_single_result(res_lstm)
    
    # GRU PyTorch
    print("Modèle GRU PyTorch...")
    res_gru = run_nn_model(df_forward_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=25, model_type='gru')
    res_gru = compute_additional_metrics(res_gru)
    plot_single_result(res_gru)
    
    # AR Model (pour comparaison)
    print("Modèle AR...")
    res_ar = run_ar_model(df_forward_agg, target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon)
    res_ar = compute_additional_metrics(res_ar)
    
    # 4. Génération des graphiques Mode 2 (Comparaison Globale)
    print("\n--- Génération de la comparaison globale (Mode 2) ---")
    results_list = [res_ar, res_arima, res_rf, res_svr, res_lstm, res_gru]
    plot_comparison(results_list)
    
    # 5. Génération du graphique Mode 3 (Cross-Link) avec GRU
    print("\n--- Génération du Cross-Link (Mode 3) ---")
    res_tx = run_nn_model(df_forward_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=20, model_type='gru')
    res_rx = run_nn_model(df_return_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=20, model_type='gru')
    plot_cross_link(res_tx, res_rx)
    
    # 6. Génération du graphique Mode 4 (Cross-Beam) avec GRU
    print("\n--- Génération du Cross-Beam (Mode 4) ---")
    beams_results = []
    for b_idx in [1, 2, 3]:
        res_b = run_nn_model(df_forward_agg, target_col=f"beam_{b_idx}", train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=20, model_type='gru')
        res_b['beam_name'] = f"beam_{b_idx}"
        beams_results.append(res_b)
    plot_cross_beam(beams_results)
    
    # 7. Génération du graphique Mode 5 (Monte Carlo)
    print("\n--- Génération du Monte Carlo (Mode 5) ---")
    models_to_test = {
        'AR/MA': lambda df, b: run_ar_model(df, target_col=b, train_frac=0.8, lags=10, horizon=horizon),
        'ARIMA(5,1,0)': lambda df, b: run_arima_model(df, target_col=b, train_frac=0.8, order=(5,1,0), horizon=horizon),
        'Random Forest': lambda df, b: run_rf_model(df, target_col=b, train_frac=0.8, seq_length=seq_len, horizon=horizon),
        'SVM': lambda df, b: run_svm_model(df, target_col=b, train_frac=0.8, seq_length=seq_len, horizon=horizon),
        'GRU': lambda df, b: run_nn_model(df, target_col=b, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=10, model_type='gru'),
        'LSTM': lambda df, b: run_nn_model(df, target_col=b, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=10, model_type='lstm')
    }
    
    mc_results = {name: {'mae': [], 'rmse': [], 'mape': [], 'r2': []} for name in models_to_test.keys()}
    noise_std = 0.5
    
    print("Démarrage des runs Monte Carlo (50 simulations)...")
    for run in range(50):
        df_noise = df_forward_agg.copy()
        df_noise[target_beam] = df_noise[target_beam] + np.random.normal(0, noise_std, len(df_noise))
        df_noise[target_beam] = np.maximum(df_noise[target_beam], 0)
        
        for name, run_fn in models_to_test.items():
            try:
                res = run_fn(df_noise, target_beam)
                res = compute_additional_metrics(res)
                mc_results[name]['mae'].append(res['mae'])
                mc_results[name]['rmse'].append(res['rmse'])
                mc_results[name]['mape'].append(res['mape'])
                mc_results[name]['r2'].append(res['r2'])
            except Exception as e:
                pass
                
    plot_monte_carlo(mc_results)
    print("\nGraphiques générés et harmonisés avec succès dans le dossier 'results' !")

if __name__ == "__main__":
    main()
