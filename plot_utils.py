import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_results_dir():
    os.makedirs('results', exist_ok=True)

def plot_single_result(result: dict):
    ensure_results_dir()
    model_name = result['model_name']
    true_values = result['true_values']
    predictions = result['predictions']

    plt.figure(figsize=(10, 6))
    
    # Sécurisation de la dimension
    true_values = np.array(true_values).flatten()
    predictions = np.array(predictions).flatten()
    
    # Trace la "Série réelle" en noir, en trait plein pour l'ancrage visuel
    plt.plot(true_values, label='Série réelle (Vérité terrain)', color='black', linewidth=2)
    plt.plot(predictions, label=f'Prédictions ({model_name})', color='orange', linestyle='--', alpha=0.8)
    plt.title(f'Prédiction vs Réalité - {model_name}')
    plt.xlabel('Pas de temps (Test Set)')
    plt.ylabel('Trafic Agrégé (Mbps)')
    plt.legend()
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = f"results/single_{model_name.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(filename)
    plt.show()

def plot_comparison(results_list: list):
    ensure_results_dir()
    
    if not results_list:
        print("Aucun résultat à afficher.")
        return

    # 1. Plot superposé des séries temporelles
    plt.figure(figsize=(12, 6))
    
    # Recherche d'un true_values non vide (au cas où le 1er modèle a échoué silencieusement)
    true_values = np.array([])
    for r in results_list:
        if len(r['true_values']) > 0:
            true_values = np.array(r['true_values']).flatten()
            break
            
    # PRINT DE DÉBOGAGE OBLIGATOIRE
    print("\n--- [DEBUG GRAPH] ---")
    print(f"Série réelle (5 premières) : {true_values[:5] if len(true_values) > 0 else 'VIDE'}")
    if len(results_list) > 0:
        first_valid_preds = np.array(results_list[-1]['predictions']).flatten()
        print(f"Prédictions du modèle {results_list[-1]['model_name']} (5 premières) : {first_valid_preds[:5]}")
    print("---------------------\n")
    
    if len(true_values) > 0:
        plt.plot(true_values, label='Série réelle (Vérité terrain)', color='black', linewidth=2)
    else:
        print("ATTENTION: La série réelle est complètement vide pour tous les modèles !")
    
    for res in results_list:
        preds = np.array(res['predictions']).flatten()
        if len(preds) > 0:
            plt.plot(preds, label=res['model_name'], linestyle='--', alpha=0.8)
        
    plt.title('Prédiction vs Réalité - Comparaison des Modèles')
    plt.xlabel('Pas de temps (Test Set)')
    plt.ylabel('Trafic Agrégé (Mbps)')
    plt.legend()
    plt.autoscale(axis='y')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('results/comparison_timeseries.png')
    plt.show()
    
    # 2. Bar chart pour la MAE
    names = [r['model_name'] for r in results_list]
    maes = [r['mae'] for r in results_list]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, maes, color='skyblue')
    plt.title('Comparaison de l\'Erreur Moyenne Absolue (MAE)')
    plt.ylabel('MAE (Mbps)')
    plt.xticks(rotation=45)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig('results/comparison_mae.png')
    plt.show()

    # 3. Bar chart pour le temps d'exécution
    plot_execution_times(results_list)

def plot_execution_times(results_list: list):
    ensure_results_dir()
    if not results_list:
        return
        
    names = [r['model_name'] for r in results_list]
    times = [r['execution_time'] for r in results_list]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, times, color='lightgreen')
    plt.title('Comparaison des Temps d\'Exécution')
    plt.ylabel('Temps (Secondes)')
    plt.xticks(rotation=45)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig('results/comparison_execution_time.png')
    plt.show()

def plot_cross_link(result_tx: dict, result_rx: dict):
    ensure_results_dir()
    model_name = result_tx['model_name']
    
    plt.figure(figsize=(12, 6))
    
    preds_tx = np.array(result_tx['predictions']).flatten()
    preds_rx = np.array(result_rx['predictions']).flatten()
    
    plt.plot(preds_tx, label='Prédictions Aller (TX - lisse)', color='blue', linestyle='-', alpha=0.8)
    plt.plot(preds_rx, label='Prédictions Retour (RX - bruité)', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'Analyse Croisée Voies (Cross-Link) - {model_name}')
    plt.xlabel('Pas de temps (Test Set)')
    plt.ylabel('Trafic Agrégé (Mbps)')
    plt.legend()
    plt.autoscale(axis='y')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/cross_link_analysis.png')
    plt.show()

def plot_cross_beam(results_beams: list):
    ensure_results_dir()
    if not results_beams:
        return
        
    model_name = results_beams[0]['model_name']
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'green', 'orange']
    for i, res in enumerate(results_beams):
        preds = np.array(res['predictions']).flatten()
        beam_name = res.get('beam_name', f'Beam {i+1}')
        plt.plot(preds, label=f'{beam_name}', color=colors[i % len(colors)], alpha=0.8)
        
    plt.title(f'Analyse Croisée Faisceaux (Cross-Beam) - {model_name}')
    plt.xlabel('Pas de temps (Test Set)')
    plt.ylabel('Trafic Agrégé (Mbps)')
    plt.legend()
    plt.autoscale(axis='y')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/cross_beam_analysis.png')
    plt.show()

