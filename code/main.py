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
from plot_utils import plot_single_result, plot_comparison, plot_cross_link, plot_cross_beam, plot_monte_carlo, plot_exhaustive_benchmark
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
    print("  5 - Analyse Monte Carlo (Évaluation de la robustesse stochastique en N runs)")
    print("  6 - Recherche Exhaustive (Benchmark global de toutes les configurations et modèles)")
    exec_mode = get_user_choice("Votre choix (1 à 6) : ", [1, 2, 3, 4, 5, 6])
    
    # Configuration en fonction du mode
    link_type = None
    target_beam = None
    
    # Étape 1 : Choix de la Voie (si pas mode 3 et pas mode 6)
    if exec_mode != 3 and exec_mode != 6:
        print("\n1. Configuration du Trafic")
        print("  1 - Voie Aller (Forward Link) : trafic descendant, lisse")
        print("  2 - Voie Retour (Return Link) : trafic montant, fragmenté")
        link_choice = get_user_choice("Votre choix (1 ou 2) : ", [1, 2])
        link_type = 'forward' if link_choice == 1 else 'return'
    
    # Étape 2 : Choix du Faisceau (si pas mode 4 et pas mode 6)
    if exec_mode != 4 and exec_mode != 6:
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
    
    selected_scenario = None
    if exec_mode != 6:
        # Étape 2.bis : Choix du Scénario (Dataset)
        print("\n2.bis Sélection du Scénario de Données")
        print("  Scénarios disponibles (nb_variable_utilisateurs) :")
        for s in range(1, 7):
            print(f"  {s} - Scénario {s}")
        scenario_choice = get_user_choice("Quel scénario charger ? (1 à 6) : ", list(range(1, 7)))
        selected_scenario = f"scenario{scenario_choice}"
    
    # Chargement des données
    def load_and_agg(l_type, current_path=None, scenario=None):
        filename = "tx_throughput.csv" if l_type == 'forward' else "rx_throughput.csv"
        scen = scenario if scenario else selected_scenario
        if current_path is None:
            current_path = os.path.join("..", "PRED_TRAFFIC", "nb_variable_utilisateurs", scen, filename)
        
        print(f"\nChargement des données ({filename}) pour {scen}...")
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
    elif exec_mode != 6:
        dfs[link_type] = load_and_agg(link_type)
        
    selected_freq = None
    dfs_agg = {}
    if exec_mode != 6:
        # Étape 3 : Fréquence d'agrégation temporelle
        print("\n3. Choix de la fréquence d'agrégation temporelle")
        print("  1 - 1 seconde (aucune agrégation supplémentaire)")
        print("  2 - 1 minute")
        print("  3 - 10 minutes")
        
        freq_choice = get_user_choice("Votre choix (1, 2 ou 3) : ", [1, 2, 3])
        freq_map = {1: '1s', 2: '1min', 3: '10min'}
        selected_freq = freq_map[freq_choice]
        
        print(f"\nAgrégation des données avec Pandas à la fréquence : {selected_freq}")
        for key, df in dfs.items():
            dfs_agg[key] = aggregate_data(df, selected_freq)
            if len(dfs_agg[key]) < 50:
                print(f"! Attention : le jeu de données agrégé pour {key} est trop petit pour de bons apprentissages !")
    
    horizon = None
    if exec_mode != 6:
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
        seq_len = 5
        if exec_mode != 6:
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
                
        elif exec_mode == 5:
            # Mode Monte Carlo
            print("\n5. Configuration de l'Analyse Monte Carlo")
            num_runs = get_user_choice("Combien de simulations Monte Carlo effectuer ? (Recommandé: 10 à 50) : ", list(range(2, 200)))
            
            # Saisie simplifiée du niveau de bruit
            print("\nNiveau de perturbation (bruit gaussien) :")
            print("  1 - Faible (2 % de l'écart-type)")
            print("  2 - Moyen (5 % de l'écart-type)")
            print("  3 - Élevé (10 % de l'écart-type)")
            noise_choice = get_user_choice("Votre choix (1, 2 ou 3) : ", [1, 2, 3])
            noise_map = {1: 0.02, 2: 0.05, 3: 0.10}
            noise_pct = noise_map[noise_choice]
            
            # Alerte si grand dataset
            num_points = len(dfs_agg[link_type])
            if num_points > 500 and num_runs > 10:
                print(f"\n⚠️ ATTENTION : Le dataset contient {num_points} points.")
                print(f"L'exécution de {num_runs} simulations avec les modèles LSTM/GRU risque de prendre plusieurs minutes.")
                confirm = input("Voulez-vous continuer ? (o/n) : ").strip().lower()
                if confirm != 'o':
                    num_runs = get_user_choice("Entrez un nombre de runs réduit (ex: 5) : ", list(range(2, 100)))
            
            print(f"\nLancement de la simulation de Monte Carlo ({num_runs} exécutions)...")
            
            mc_results = {
                'AR/MA': {'mae': [], 'rmse': [], 'mape': [], 'r2': []},
                'ARIMA': {'mae': [], 'rmse': [], 'mape': [], 'r2': []},
                'Random Forest': {'mae': [], 'rmse': [], 'mape': [], 'r2': []},
                'SVM': {'mae': [], 'rmse': [], 'mape': [], 'r2': []},
                'GRU PyTorch': {'mae': [], 'rmse': [], 'mape': [], 'r2': []},
                'LSTM PyTorch': {'mae': [], 'rmse': [], 'mape': [], 'r2': []}
            }
            
            base_df = dfs_agg[link_type]
            std_val = base_df[target_beam].std()
            
            for run in range(1, num_runs + 1):
                print(f"\n--- Simulation Monte Carlo {run}/{num_runs} ---")
                
                # Génération du bruit gaussien
                perturbed_df = base_df.copy()
                np.random.seed(run)
                noise = np.random.normal(0, std_val * noise_pct, size=len(base_df))
                perturbed_df[target_beam] = np.maximum(perturbed_df[target_beam] + noise, 0)
                
                # 1. AR/MA
                try:
                    res = run_ar_model(perturbed_df, target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon)
                    res = compute_additional_metrics(res)
                    mc_results['AR/MA']['mae'].append(res['mae'])
                    mc_results['AR/MA']['rmse'].append(res['rmse'])
                    mc_results['AR/MA']['mape'].append(res['mape'])
                    mc_results['AR/MA']['r2'].append(res['r2'])
                except Exception as e: print(f"  -> Erreur AR : {e}")
                
                # 2. ARIMA
                try:
                    res = run_arima_model(perturbed_df, target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon)
                    res = compute_additional_metrics(res)
                    mc_results['ARIMA']['mae'].append(res['mae'])
                    mc_results['ARIMA']['rmse'].append(res['rmse'])
                    mc_results['ARIMA']['mape'].append(res['mape'])
                    mc_results['ARIMA']['r2'].append(res['r2'])
                except Exception as e: print(f"  -> Erreur ARIMA : {e}")
                
                # 3. RF
                try:
                    res = run_rf_model(perturbed_df, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
                    res = compute_additional_metrics(res)
                    mc_results['Random Forest']['mae'].append(res['mae'])
                    mc_results['Random Forest']['rmse'].append(res['rmse'])
                    mc_results['Random Forest']['mape'].append(res['mape'])
                    mc_results['Random Forest']['r2'].append(res['r2'])
                except Exception as e: print(f"  -> Erreur RF : {e}")
                
                # 4. SVM
                try:
                    res = run_svm_model(perturbed_df, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
                    res = compute_additional_metrics(res)
                    mc_results['SVM']['mae'].append(res['mae'])
                    mc_results['SVM']['rmse'].append(res['rmse'])
                    mc_results['SVM']['mape'].append(res['mape'])
                    mc_results['SVM']['r2'].append(res['r2'])
                except Exception as e: print(f"  -> Erreur SVM : {e}")
                
                # 5. GRU
                try:
                    res = run_nn_model(perturbed_df, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='gru')
                    res = compute_additional_metrics(res)
                    mc_results['GRU PyTorch']['mae'].append(res['mae'])
                    mc_results['GRU PyTorch']['rmse'].append(res['rmse'])
                    mc_results['GRU PyTorch']['mape'].append(res['mape'])
                    mc_results['GRU PyTorch']['r2'].append(res['r2'])
                except Exception as e: print(f"  -> Erreur GRU : {e}")
                
                # 6. LSTM
                try:
                    res = run_nn_model(perturbed_df, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='lstm')
                    res = compute_additional_metrics(res)
                    mc_results['LSTM PyTorch']['mae'].append(res['mae'])
                    mc_results['LSTM PyTorch']['rmse'].append(res['rmse'])
                    mc_results['LSTM PyTorch']['mape'].append(res['mape'])
                    mc_results['LSTM PyTorch']['r2'].append(res['r2'])
                except Exception as e: print(f"  -> Erreur LSTM : {e}")
                
            # Affichage de la synthèse statistique
            print("\n" + "=" * 60)
            print("       SYNTHÈSE STATISTIQUE DES SIMULATIONS MONTE CARLO       ")
            print("=" * 60)
            for model_name, metrics in mc_results.items():
                if len(metrics['mae']) > 0:
                    print(f"\nModèle : {model_name}")
                    print(f"  MAE  : Moyenne = {np.mean(metrics['mae']):.4f} Mbps, Écart-Type = {np.std(metrics['mae']):.4f}")
                    print(f"  RMSE : Moyenne = {np.mean(metrics['rmse']):.4f} Mbps, Écart-Type = {np.std(metrics['rmse']):.4f}")
                    print(f"  MAPE : Moyenne = {np.mean(metrics['mape'])*100:.2f} %, Écart-Type = {np.std(metrics['mape'])*100:.2f} %")
                    print(f"  R²   : Moyenne = {np.mean(metrics['r2']):.4f}, Écart-Type = {np.std(metrics['r2']):.4f}")
            print("=" * 60)
            
            # Génération des graphiques Boxplot
            plot_monte_carlo(mc_results)
            
        elif exec_mode == 6:
            # Mode Recherche Exhaustive (Grid-Search)
            print("\n" + "=" * 60)
            print("         LANCEMENT DU BENCHMARK EXHAUSTIF SUR TOUS LES SCÉNARIOS         ")
            print("=" * 60)
            print("Ce mode teste tous les modèles sur toutes les configurations possibles de tous les scénarios :")
            print("  - 6 Scénarios de trafic (scenario1 à scenario6)")
            print("  - 2 Voies de trafic (TX/Aller et RX/Retour)")
            print("  - 3 Fréquences d'agrégation (1s, 1min, 10min)")
            print("  - 3 Faisceaux satellites (beam_1, beam_2, beam_3)")
            print("  - 2 Horizons de prédiction (H=1 et H=5)")
            print("Total de 6 scénarios x 36 configurations = 216 configurations physiques uniques.")
            print("Total de 216 configs x 6 modèles = 1296 runs de prédiction.")
            print("Optimisations de vitesse activées (NN époques=10, 1s tronqué à 1500 points).")
            print("-" * 60)
            
            benchmark_results = []
            
            # Liste des dimensions à parcourir
            scenarios = [f"scenario{i}" for i in range(1, 7)]
            links = ['forward', 'return']
            frequencies = ['1s', '1min', '10min']
            beams = ['beam_1', 'beam_2', 'beam_3']
            horizons = [1, 5]
            
            run_counter = 0
            total_runs = len(scenarios) * len(links) * len(frequencies) * len(beams) * len(horizons) * 6
            
            # On désactive temporairement les warnings d'entraînement
            import warnings
            warnings.filterwarnings("ignore")
            
            for scen in scenarios:
                print(f"\n" + "=" * 60)
                print(f"       DÉBUT DE L'ÉVALUATION DU SCÉNARIO : {scen.upper()}       ")
                print("=" * 60)
                
                # Charger les données pour ce scénario spécifique
                scen_dfs = {}
                try:
                    scen_dfs['forward'] = load_and_agg('forward', scenario=scen)
                    scen_dfs['return'] = load_and_agg('return', scenario=scen)
                except Exception as load_err:
                    print(f"! Erreur de chargement pour {scen} : {load_err}")
                    continue
                
                for l_type in links:
                    for freq in frequencies:
                        # Pré-agréger les données pour cette voie et fréquence
                        print(f"\n[Agrégation] Préparation de la voie {l_type.upper()} à la fréquence {freq} ({scen})...")
                        raw_df = scen_dfs[l_type]
                        aggregated_df = aggregate_data(raw_df, freq)
                        
                        # Optimisation de vitesse critique pour la fréquence 1s
                        if freq == '1s':
                            aggregated_df = aggregated_df.iloc[:1500]
                            
                        for beam in beams:
                            for h in horizons:
                                # Calcul de la longueur de séquence
                                s_len = 5 # Fixé à 5 pour homogénéité et vitesse du benchmark
                                
                                # Définir l'évaluation pour chaque modèle
                                models_fns = {
                                    'AR/MA': lambda df, b, h: run_ar_model(df, target_col=b, train_frac=0.8, lags=10, horizon=h),
                                    'ARIMA(5,1,0)': lambda df, b, h: run_arima_model(df, target_col=b, train_frac=0.8, order=(5,1,0), horizon=h),
                                    'Random Forest': lambda df, b, h: run_rf_model(df, target_col=b, train_frac=0.8, seq_length=s_len, horizon=h),
                                    'SVM': lambda df, b, h: run_svm_model(df, target_col=b, train_frac=0.8, seq_length=s_len, horizon=h),
                                    'GRU PyTorch': lambda df, b, h: run_nn_model(df, target_col=b, train_frac=0.8, seq_length=s_len, horizon=h, epochs=10, model_type='gru'),
                                    'LSTM PyTorch': lambda df, b, h: run_nn_model(df, target_col=b, train_frac=0.8, seq_length=s_len, horizon=h, epochs=10, model_type='lstm')
                                }
                                
                                print(f"\n---> Config [{scen.upper()}]: Voie={l_type.upper()}, Freq={freq}, Beam={beam}, Horizon={h} <---")
                                
                                for m_name, run_fn in models_fns.items():
                                    run_counter += 1
                                    print(f"  [{run_counter}/{total_runs}] Modèle: {m_name}...", end="", flush=True)
                                    try:
                                        res = run_fn(aggregated_df, beam, h)
                                        if res:
                                            res = compute_additional_metrics(res)
                                            
                                            # Stocker les métriques avec les métadonnées de la configuration et du scénario
                                            record = {
                                                'scenario': scen,
                                                'link_type': l_type,
                                                'freq': freq,
                                                'beam_name': beam,
                                                'horizon': h,
                                                'model_name': m_name,
                                                'mae': res['mae'],
                                                'rmse': res['rmse'],
                                                'mape': res['mape'],
                                                'r2': res['r2'],
                                                'execution_time': res['execution_time']
                                            }
                                            benchmark_results.append(record)
                                            print(f" OK (MAE={res['mae']:.3f}, R²={res['r2']:.3f})")
                                        else:
                                            print(" Échoué (retour vide)")
                                    except Exception as ex:
                                        print(f" Erreur : {ex}")
                                        
            # Sauvegarde des résultats sous forme de fichier CSV pour exploitation ultérieure
            if benchmark_results:
                import pandas as pd
                df_bench = pd.DataFrame(benchmark_results)
                df_bench.to_csv("results/exhaustive_benchmark.csv", index=False, sep=";")
                print("\n" + "=" * 60)
                print("Recherche exhaustive sur tous les scénarios terminée avec succès !")
                print("Résultats bruts sauvegardés dans : results/exhaustive_benchmark.csv")
                
                # Génération et tracé du graphique de synthèse (Heatmap R² + Win Count)
                print("Génération du graphique comparatif global...")
                plot_exhaustive_benchmark(benchmark_results)
                print("Graphique sauvegardé dans : results/exhaustive_benchmark.png")
                print("=" * 60)
            else:
                print("\nAucun run n'a produit de résultat exploitable.")
                
    except Exception as e:
        print(f"\nUne erreur est survenue lors de l'exécution : {e}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt par l'utilisateur.")
        sys.exit(0)
