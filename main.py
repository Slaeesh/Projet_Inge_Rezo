import os
import sys

from data_generator import generate_fake_traffic_data
from data_aggregator import aggregate_data
from models.ar_model import run_ar_model
from models.nn_model import run_nn_model

def get_user_choice(prompt, options):
    while True:
        try:
            choice = int(input(prompt))
            if choice in options:
                return choice
            else:
                print(f"Veuillez choisir une option valide : {list(options.keys)}")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

def main():
    print("=" * 50)
    print("Outil de Prédiction de Trafic Réseau par Satellite")
    print("=" * 50)
    
    # Étape 1 : Génération des données
    print("\n1. Génération des données de trafic")
    print("Simulation du trafic avec un cycle journalier et du bruit (en attendant ns-2)...")
    df = generate_fake_traffic_data(duration_hours=48)
    print(f"Jeu de données généré (résolution 1s) : {len(df)} lignes.")
    
    # Étape 2 : Choix de la fréquence d'agrégation
    print("\n2. Choix de la fréquence d'agrégation temporelle")
    print("  1 - 1 seconde (aucune agrégation supplémentaire)")
    print("  2 - 1 minute")
    print("  3 - 10 minutes")
    
    freq_choice = get_user_choice("Votre choix (1, 2 ou 3) : ", [1, 2, 3])
    freq_map = {1: '1s', 2: '1min', 3: '10min'}
    selected_freq = freq_map[freq_choice]
    
    print(f"\nAgrégation des données avec Pandas à la fréquence : {selected_freq}")
    df_agg = aggregate_data(df, selected_freq)
    print(f"Jeu de données agrégé : {len(df_agg)} lignes associées.")
    
    if len(df_agg) < 50:
        print("Attention : le jeu de données agrégé est très petit, les modèles risquent de ne pas bien s'entraîner.")
        print("Diminuez l'intervalle d'agrégation ou augmentez la durée de génération.")
    
    # Étape 3 : Choix du modèle
    print("\n3. Choix du modèle prédictif")
    print("  1 - Modèle Statistique (Autorégressif - AR)")
    print("  2 - Modèle Réseaux de Neurones (MLP - PyTorch)")
    
    model_choice = get_user_choice("Votre choix (1 ou 2) : ", [1, 2])
    
    # Étape 4 : Exécution
    print("\nLancement de l'entraînement et de la prédiction...\n" + "-" * 30)
    
    try:
        if model_choice == 1:
            # Modèle AR
            results = run_ar_model(df_agg, train_frac=0.8, lags=10)
        else:
            # Modèle NN
            # On adapte la longueur de séquence en fonction du nombre de données dispos
            seq_len = min(10, max(1, len(df_agg) // 20))
            if seq_len < 1:
                seq_len = 1
            results = run_nn_model(df_agg, train_frac=0.8, seq_length=seq_len, epochs=50)

        # Affichage des résultats finaux (Complexité et Précision)
        print("\n--- RÉSULTATS ---")
        print(f"Modèle utilisé  : {results['model_name']}")
        print(f"Temps exécution : {results['execution_time']:.4f} secondes (Complexité algorithmique / Wall time)")
        print(f"Erreur Moyenne  : {results['mae']:.4f} Mbps (Précision MAE absolue)")
        print("-----------------")
        
    except Exception as e:
        print(f"\nUne erreur est survenue lors de l'exécution du modèle : {e}")

if __name__ == '__main__':
    # Empêche quelques warnings de s'afficher dans la belle interface CLI
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt par l'utilisateur.")
        sys.exit(0)
