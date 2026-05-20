import os
import sys
import warnings

from data_generator import generate_fake_traffic_data, load_real_traffic_data
from data_aggregator import aggregate_data
from models.ar_model import run_ar_model
from models.arima_model import run_arima_model
from models.nn_model import run_nn_model
from models.rf_model import run_rf_model
from models.svm_model import run_svm_model
from plot_utils import plot_single_result, plot_comparison, plot_cross_link, plot_cross_beam
import numpy as np

def compute_additional_metrics(res):
    """
    Calcule de manière centralisée les métriques RMSE, MAPE et R² 
    pour éviter de modifier le code de chaque modèle.
    """
    if not res:
        return res
    
    true = np.array(res['true_values']).flatten()
    pred = np.array(res['predictions']).flatten()
    
    if len(true) == 0 or len(pred) == 0:
        res['rmse'] = 0.0
        res['mape'] = 0.0
        res['r2'] = 0.0
        return res
        
    # 1. RMSE
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    
    # 2. MAPE (sécurisé contre les divisions par zéro)
    mape = np.mean(np.abs((true - pred) / np.clip(true, 1e-5, None)))
    
    # 3. R² (Coefficient de détermination)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    res['rmse'] = rmse
    res['mape'] = mape
    res['r2'] = r2
    return res

def get_user_choice(prompt: str, options: list):
    while True:
        try:
            choice = int(input(prompt))
            if choice in options:
                return choice
            else:
                print(f"Veuillez choisir une option valide : {options}")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

def get_model_choice():
    print("\nArchitecture Prédictive : Modèles")
    print("  1 - AR/MA (Statistique AutoRégressif)")
    print("  2 - ARIMA (Statistique Intégré)")
    print("  3 - Random Forest (Machine Learning)")
    print("  4 - SVM (Support Vector Machine)")
    print("  5 - GRU (Deep Learning PyTorch)")
    print("  6 - LSTM (Deep Learning PyTorch)")
    return get_user_choice("Votre choix (1 à 6) : ", [1, 2, 3, 4, 5, 6])

def run_selected_model(model_choice, df_agg, target_beam, seq_len, horizon):
    if model_choice == 1:
        return run_ar_model(df_agg, target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon)
    elif model_choice == 2:
        return run_arima_model(df_agg, target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon)
    elif model_choice == 3:
        return run_rf_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
    elif model_choice == 4:
        return run_svm_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
    elif model_choice == 5:
        return run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='gru')
    elif model_choice == 6:
        return run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='lstm')
    return None

def main():
    print("=" * 60)
    print(" Outil de Prédiction de Trafic Réseau par Satellite ")
    print("=" * 60)
    
    # Mode d'exécution
    print("\n0. Mode d'exécution")
    print("  1 - Analyse Classique (1 modèle avec graphique simple)")
    print("  2 - Comparaison Globale (Couverture de tous les modèles avec graphiques multiples)")
    print("  3 - Analyse Croisée Voies (Cross-Link : TX vs RX sur 1 faisceau avec 1 modèle)")
    print("  4 - Analyse Croisée Faisceaux (Cross-Beam : 3 faisceaux simultanés sur 1 voie avec 1 modèle)")
    exec_mode = get_user_choice("Votre choix (1 à 4) : ", [1, 2, 3, 4])
    
    # Configuration en fonction du mode
    link_type = None
    target_beam = None
    
    # Étape 1 : Choix de la Voie (si pas mode 3)
    if exec_mode != 3:
        print("\n1. Configuration du Trafic")
        print("  1 - Voie Aller (Forward Link) : trafic descendant, lisse")
        print("  2 - Voie Retour (Return Link) : trafic montant, fragmenté")
        link_choice = get_user_choice("Votre choix (1 ou 2) : ", [1, 2])
        link_type = 'forward' if link_choice == 1 else 'return'
    
    # Étape 2 : Choix du Faisceau (si pas mode 4)
    if exec_mode != 4:
        print("\n2. Sélection du Faisceau (Dimension Spatiale)")
        print("  Profils des faisceaux :")
        print("  Beam 1 = Zone Dense urbaine (Charge élevée, forte variance).")
        print("  Beam 2 = Zone Rurale (Charge faible, relativement constant).")
        print("  Beam 3 = Zone Intermédiaire (Charge moyenne).")
        
        num_beams = 3
        options_beams = list(range(1, num_beams + 1))
        for b in options_beams:
            print(f"  {b} - beam_{b}")
        beam_choice = get_user_choice(f"Quel faisceau cibler ? {options_beams} : ", options_beams)
        target_beam = f'beam_{beam_choice}'
    
    # Chargement des données
    def load_and_agg(l_type, current_path=None):
        filename = "tx_throughput.csv" if l_type == 'forward' else "rx_throughput.csv"
        if current_path is None:
            current_path = os.path.join("..", "PRED_TRAFFIC", "nb_variable_utilisateurs", "scenario1", filename)
        
        print(f"\nChargement des données ({filename})...")
        while True:
            try:
                df = load_real_traffic_data(l_type, current_path)
                print(f"[{'Voie Aller' if l_type == 'forward' else 'Voie Retour'}] Données chargées avec succès : {len(df)} lignes, 3 faisceaux.")
                return df
            except FileNotFoundError:
                print(f"! Erreur : Le fichier est introuvable au chemin : {current_path}")
                current_path = input(f"Veuillez saisir le chemin absolu (ou relatif) vers le fichier {filename} : ").strip()
            except IsADirectoryError:
                print(f"! Erreur : Vous avez saisi le chemin d'un dossier. Vous devez indiquer le chemin jusqu'au fichier.")
                current_path = input(f"Veuillez saisir le chemin complet vers le fichier {filename} : ").strip()
            except Exception as e:
                print(f"! Erreur inattendue lors de la lecture du fichier : {e}")
                sys.exit(1)

    dfs = {}
    if exec_mode == 3:
        dfs['forward'] = load_and_agg('forward')
        dfs['return'] = load_and_agg('return')
    else:
        dfs[link_type] = load_and_agg(link_type)
        
    # Étape 3 : Fréquence d'agrégation temporelle
    print("\n3. Choix de la fréquence d'agrégation temporelle")
    print("  1 - 1 seconde (aucune agrégation supplémentaire)")
    print("  2 - 1 minute")
    print("  3 - 10 minutes")
    
    freq_choice = get_user_choice("Votre choix (1, 2 ou 3) : ", [1, 2, 3])
    freq_map = {1: '1s', 2: '1min', 3: '10min'}
    selected_freq = freq_map[freq_choice]
    
    print(f"\nAgrégation des données avec Pandas à la fréquence : {selected_freq}")
    dfs_agg = {}
    for key, df in dfs.items():
        dfs_agg[key] = aggregate_data(df, selected_freq)
        if len(dfs_agg[key]) < 50:
            print(f"! Attention : le jeu de données agrégé pour {key} est trop petit pour de bons apprentissages !")
    
    # Étape 4 : Horizon de prédiction
    print("\n4. Horizon de prédiction")
    print("Combien de pas de temps dans le futur souhaitez-vous prédire ? (ex: 1 = pas suivant)")
    horizon = get_user_choice("Votre choix (entier positif, ex: 1, 3, 5..) : ", list(range(1, 100)))
    
    # Étape 5 : Choix du Modèle
    model_choice = None
    if exec_mode in [1, 3, 4]:
        model_choice = get_model_choice()
        
    print("\n" + "-" * 40)
    print("Lancement de l'entraînement et de la prédiction...")
    print("-" * 40)
    
    try:
        # We use dfs_agg[list(dfs_agg.keys())[0]] to get the first df for seq_len calculation
        sample_df = list(dfs_agg.values())[0]
        seq_len = min(10, max(1, len(sample_df) // 20))
        if seq_len < 1: seq_len = 1
        
        def display_single_result(res, beam, freq):
            print("\n" + "=" * 40)
            print("               RÉSULTATS               ")
            print("=" * 40)
            print(f"Modèle utilisé          : {res['model_name']}")
            print(f"Faisceau / Beam         : {beam}")
            print(f"Horizon temporel        : {res['horizon']} ({freq} en avant)")
            print(f"Temps exécution         : {res['execution_time']:.4f} sec")
            print(f"Erreur Moyenne (MAE)    : {res['mae']:.4f} Mbps")
            
            if 'rmse' in res and 'mape' in res:
                print(f"RMSE                    : {res['rmse']:.4f} Mbps")
                print(f"MAPE                    : {res['mape']:.2%}")
                print(f"R² (Score)              : {res.get('r2', 0.0):.4f}")
                
                mape = res['mape']
                if mape < 0.10:
                    diag = "Prédiction très fiable (adaptée pour une allocation dynamique agressive)."
                elif mape <= 0.20:
                    diag = "Prédiction acceptable (adaptée pour une allocation de sécurité macroscopique)."
                else:
                    diag = "Prédiction non pertinente / rejetée pour la production."
                print(f"Diagnostic (MAPE)       : {diag}")
                
            print("-" * 40)
            print("Métriques Métiers (Allocation) :")
            print(f" - Taux de Sous-allocation : {res['under_allocation']:.2f} %  -> Risque de Congestion")
            print(f" - Taux de Sur-allocation  : {res['over_allocation']:.2f} %  -> Gaspillage de ressources")
            print("=" * 40)

        if exec_mode == 1:
            results = run_selected_model(model_choice, dfs_agg[link_type], target_beam, seq_len, horizon)
            if results:
                results = compute_additional_metrics(results)
                display_single_result(results, target_beam, selected_freq)
                plot_single_result(results)
                
        elif exec_mode == 2:
            all_results = []
            print("[1/6] Exécution du Modèle AR/MA...")
            try: all_results.append(run_ar_model(dfs_agg[link_type], target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon))
            except Exception as e: print(f"  -> Erreur AR : {e}")
            
            print("[2/6] Exécution du Modèle ARIMA...")
            try: all_results.append(run_arima_model(dfs_agg[link_type], target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon))
            except Exception as e: print(f"  -> Erreur ARIMA : {e}")
            
            print("[3/6] Exécution du Modèle Random Forest...")
            try: all_results.append(run_rf_model(dfs_agg[link_type], target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon))
            except Exception as e: print(f"  -> Erreur RF : {e}")
            
            print("[4/6] Exécution du Modèle SVM...")
            try: all_results.append(run_svm_model(dfs_agg[link_type], target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon))
            except Exception as e: print(f"  -> Erreur SVM : {e}")
            
            print("[5/6] Exécution du Modèle GRU...")
            try: all_results.append(run_nn_model(dfs_agg[link_type], target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='gru'))
            except Exception as e: print(f"  -> Erreur GRU : {e}")
            
            print("[6/6] Exécution du Modèle LSTM...")
            try: all_results.append(run_nn_model(dfs_agg[link_type], target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='lstm'))
            except Exception as e: print(f"  -> Erreur LSTM : {e}")
 
            if all_results:
                # Filtrer les None et calculer les métriques additionnelles
                all_results = [compute_additional_metrics(res) for res in all_results if res is not None]
                print(f"\nTerminé ! {len(all_results)} modèles testés avec succès.")
                print("Génération et affichage des graphiques...")
                plot_comparison(all_results)
            else:
                print("\nAucun modèle n'a produit de résultat exploitable.")
                
        elif exec_mode == 3:
            # Cross-Link
            print("\nExécution sur la Voie Aller (TX)...")
            res_tx = run_selected_model(model_choice, dfs_agg['forward'], target_beam, seq_len, horizon)
            if res_tx:
                res_tx = compute_additional_metrics(res_tx)
                display_single_result(res_tx, target_beam + " (Aller)", selected_freq)
            
            print("\nExécution sur la Voie Retour (RX)...")
            res_rx = run_selected_model(model_choice, dfs_agg['return'], target_beam, seq_len, horizon)
            if res_rx:
                res_rx = compute_additional_metrics(res_rx)
                display_single_result(res_rx, target_beam + " (Retour)", selected_freq)
                
            if res_tx and res_rx:
                plot_cross_link(res_tx, res_rx)
                
        elif exec_mode == 4:
            # Cross-Beam
            results_beams = []
            for b in [1, 2, 3]:
                beam = f'beam_{b}'
                print(f"\nExécution sur le {beam}...")
                res = run_selected_model(model_choice, dfs_agg[link_type], beam, seq_len, horizon)
                if res:
                    res = compute_additional_metrics(res)
                    res['beam_name'] = f'Beam {b}'
                    display_single_result(res, beam, selected_freq)
                    results_beams.append(res)
            
            if results_beams:
                plot_cross_beam(results_beams)
                
    except Exception as e:
        print(f"\nUne erreur est survenue lors de l'exécution : {e}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt par l'utilisateur.")
        sys.exit(0)
