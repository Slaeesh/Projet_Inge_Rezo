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
from plot_utils import plot_single_result, plot_comparison

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

def main():
    print("=" * 60)
    print(" Outil de Prédiction de Trafic Réseau par Satellite ")
    print("=" * 60)
    
    # Mode d'exécution
    print("\n0. Mode d'exécution")
    print("  1 - Analyse Classique (1 modèle avec graphique simple)")
    print("  2 - Comparaison Globale (Couverture de tous les modèles avec graphiques multiples)")
    exec_mode = get_user_choice("Votre choix (1 ou 2) : ", [1, 2])
    
    # Étape 1 : Choix de la Voie
    print("\n1. Configuration du Trafic")
    print("  1 - Voie Aller (Forward Link) : trafic descendant, lisse")
    print("  2 - Voie Retour (Return Link) : trafic montant, fragmenté")
    link_choice = get_user_choice("Votre choix (1 ou 2) : ", [1, 2])
    link_type = 'forward' if link_choice == 1 else 'return'
    
    filename = "tx_throughput.csv" if link_type == 'forward' else "rx_throughput.csv"
    default_path = os.path.join("..", "PRED_TRAFFIC", "nb_variable_utilisateurs", "scenario1", filename)
    
    print(f"\nChargement des données réelles ({filename})...")
    df = None
    current_path = default_path
    
    while True:
        try:
            df = load_real_traffic_data(link_type, current_path)
            print(f"[{'Voie Aller' if link_type == 'forward' else 'Voie Retour'}] Données chargées avec succès : {len(df)} lignes, 3 faisceaux.")
            break
        except FileNotFoundError:
            print(f"! Erreur : Le fichier est introuvable au chemin : {current_path}")
            current_path = input(f"Veuillez saisir le chemin absolu (ou relatif) vers le fichier {filename} : ").strip()
        except IsADirectoryError:
            print(f"! Erreur : Vous avez saisi le chemin d'un dossier. Vous devez indiquer le chemin jusqu'au fichier (ex: .../Data/{filename}).")
            current_path = input(f"Veuillez saisir le chemin complet vers le fichier {filename} : ").strip()
        except Exception as e:
            print(f"! Erreur inattendue lors de la lecture du fichier : {e}")
            sys.exit(1)
            
    num_beams = 3
    
    # Étape 2 : Choix du Faisceau (Dimension Spatiale)
    print("\n2. Sélection du Faisceau (Dimension Spatiale)")
    options_beams = list(range(1, num_beams + 1))
    for b in options_beams:
        print(f"  {b} - beam_{b}")
    beam_choice = get_user_choice(f"Quel faisceau cibler ? {options_beams} : ", options_beams)
    target_beam = f'beam_{beam_choice}'
    
    # Étape 3 : Fréquence d'agrégation temporelle
    print("\n3. Choix de la fréquence d'agrégation temporelle")
    print("  1 - 1 seconde (aucune agrégation supplémentaire)")
    print("  2 - 1 minute")
    print("  3 - 10 minutes")
    
    freq_choice = get_user_choice("Votre choix (1, 2 ou 3) : ", [1, 2, 3])
    freq_map = {1: '1s', 2: '1min', 3: '10min'}
    selected_freq = freq_map[freq_choice]
    
    print(f"\nAgrégation des données avec Pandas à la fréquence : {selected_freq}")
    df_agg = aggregate_data(df, selected_freq)
    print(f"Jeu de données agrégé : {len(df_agg)} lignes.")
    
    if len(df_agg) < 50:
        print("! Attention : le jeu de données agrégé est trop petit pour de bons apprentissages !")
    
    # Étape 4 : Horizon de prédiction
    print("\n4. Horizon de prédiction")
    print("Combien de pas de temps dans le futur souhaitez-vous prédire ? (ex: 1 = pas suivant)")
    horizon = get_user_choice("Votre choix (entier positif, ex: 1, 3, 5..) : ", list(range(1, 100)))
    
    # Étape 5 : Choix du Modèle (Dispatch Menu)
    if exec_mode == 1:
        print("\n5. Architecture Prédictive : Modèles")
        print("  1 - AR/MA (Statistique AutoRégressif)")
        print("  2 - ARIMA (Statistique Intégré)")
        print("  3 - Random Forest (Machine Learning)")
        print("  4 - SVM (Support Vector Machine)")
        print("  5 - GRU (Deep Learning PyTorch)")
        print("  6 - LSTM (Deep Learning PyTorch)")
        
        model_choice = get_user_choice("Votre choix (1 à 6) : ", [1, 2, 3, 4, 5, 6])
    else:
        print("\n5. Comparaison Globale : Tous les algorithmes vont être exécutés sur le même jeu de données.")
        model_choice = None
    
    print("\n" + "-" * 40)
    print("Lancement de l'entraînement et de la prédiction...")
    print("-" * 40)
    
    results = None
    all_results = []
    
    try:
        seq_len = min(10, max(1, len(df_agg) // 20))
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
            print("-" * 40)
            print("Métriques Métiers (Allocation) :")
            print(f" - Taux de Sous-allocation : {res['under_allocation']:.2f} %  -> Risque de Congestion")
            print(f" - Taux de Sur-allocation  : {res['over_allocation']:.2f} %  -> Gaspillage de ressources")
            print("=" * 40)
            plot_single_result(res)

        if exec_mode == 1:
            if model_choice == 1:
                results = run_ar_model(df_agg, target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon)
            elif model_choice == 2:
                results = run_arima_model(df_agg, target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon)
            elif model_choice == 3:
                results = run_rf_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
            elif model_choice == 4:
                results = run_svm_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon)
            elif model_choice == 5:
                results = run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='gru')
            elif model_choice == 6:
                results = run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='lstm')
            
            if results:
                display_single_result(results, target_beam, selected_freq)
        else:
            # Mode Comparaison
            print("[1/6] Exécution du Modèle AR/MA...")
            try: all_results.append(run_ar_model(df_agg, target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon))
            except Exception as e: print(f"  -> Erreur AR : {e}")
            
            print("[2/6] Exécution du Modèle ARIMA...")
            try: all_results.append(run_arima_model(df_agg, target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon))
            except Exception as e: print(f"  -> Erreur ARIMA : {e}")
            
            print("[3/6] Exécution du Modèle Random Forest...")
            try: all_results.append(run_rf_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon))
            except Exception as e: print(f"  -> Erreur RF : {e}")
            
            print("[4/6] Exécution du Modèle SVM...")
            try: all_results.append(run_svm_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon))
            except Exception as e: print(f"  -> Erreur SVM : {e}")
            
            print("[5/6] Exécution du Modèle GRU...")
            try: all_results.append(run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='gru'))
            except Exception as e: print(f"  -> Erreur GRU : {e}")
            
            print("[6/6] Exécution du Modèle LSTM...")
            try: all_results.append(run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50, model_type='lstm'))
            except Exception as e: print(f"  -> Erreur LSTM : {e}")

            if all_results:
                print(f"\nTerminé ! {len(all_results)} modèles testés avec avec succès.")
                print("Génération et affichage des graphiques...")
                plot_comparison(all_results)
            else:
                print("\nAucun modèle n'a produit de résultat exploitable.")
            
    except Exception as e:
        print(f"\nUne erreur est survenue lors de l'exécution : {e}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt par l'utilisateur.")
        sys.exit(0)
