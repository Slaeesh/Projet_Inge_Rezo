import os
import sys
import warnings

from data_generator import generate_fake_traffic_data
from data_aggregator import aggregate_data
from models.ar_model import run_ar_model
from models.arima_model import run_arima_model
from models.nn_model import run_nn_model

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
    
    # Étape 1 : Choix de la Voie
    print("\n1. Configuration du Trafic")
    print("  1 - Voie Aller (Forward Link) : trafic descendant, lisse")
    print("  2 - Voie Retour (Return Link) : trafic montant, fragmenté")
    link_choice = get_user_choice("Votre choix (1 ou 2) : ", [1, 2])
    link_type = 'forward' if link_choice == 1 else 'return'
    
    print("\nSimulation du trafic en cours...")
    num_beams = 3
    df = generate_fake_traffic_data(duration_hours=48, link_type=link_type, num_beams=num_beams)
    print(f"[{'Voie Aller' if link_type == 'forward' else 'Voie Retour'}] Jeu de données généré : {len(df)} lignes, {num_beams} faisceaux.")
    
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
    print("\n5. Architecture Prédictive : Modèles")
    print("  1 - AR/MA (Statistique AutoRégressif)")
    print("  2 - ARIMA (Statistique Intégré)")
    print("  3 - Random Forest (Machine Learning)")
    print("  4 - SVM (Support Vector Machine)")
    print("  5 - RNN/MLP (Deep Learning PyTorch)")
    
    model_choice = get_user_choice("Votre choix (1 à 5) : ", [1, 2, 3, 4, 5])
    
    print("\n" + "-" * 40)
    print("Lancement de l'entraînement et de la prédiction...")
    print("-" * 40)
    
    results = None
    try:
        # Dispatcher local pour exécuter le module adéquat
        if model_choice == 1:
            results = run_ar_model(df_agg, target_col=target_beam, train_frac=0.8, lags=10, horizon=horizon)
        elif model_choice == 2:
            results = run_arima_model(df_agg, target_col=target_beam, train_frac=0.8, order=(5, 1, 0), horizon=horizon)
        elif model_choice == 3:
            print("Modèle Random Forest en cours d'intégration...")
        elif model_choice == 4:
            print("Modèle SVM en cours d'intégration...")
        elif model_choice == 5:
            seq_len = min(10, max(1, len(df_agg) // 20))
            if seq_len < 1: seq_len = 1
            results = run_nn_model(df_agg, target_col=target_beam, train_frac=0.8, seq_length=seq_len, horizon=horizon, epochs=50)
        
        if results:
            print("\n" + "=" * 40)
            print("               RÉSULTATS               ")
            print("=" * 40)
            print(f"Modèle utilisé          : {results['model_name']}")
            print(f"Faisceau / Beam         : {target_beam}")
            print(f"Horizon temporel        : {results['horizon']} ({selected_freq} en avant)")
            print(f"Temps exécution         : {results['execution_time']:.4f} sec")
            print(f"Erreur Moyenne (MAE)    : {results['mae']:.4f} Mbps")
            print("-" * 40)
            print("Métriques Métiers (Allocation) :")
            print(f" - Taux de Sous-allocation : {results['under_allocation']:.2f} %  -> Risque de Congestion")
            print(f" - Taux de Sur-allocation  : {results['over_allocation']:.2f} %  -> Gaspillage de ressources")
            print("=" * 40)
            
    except Exception as e:
        print(f"\nUne erreur est survenue lors de l'exécution : {e}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt par l'utilisateur.")
        sys.exit(0)
