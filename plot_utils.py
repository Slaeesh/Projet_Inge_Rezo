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

    # 3. Bar chart pour la RMSE (Root Mean Squared Error)
    rmses = [r.get('rmse', 0.0) for r in results_list]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, rmses, color='salmon')
    plt.title('Comparaison de la Root Mean Squared Error (RMSE)')
    plt.ylabel('RMSE (Mbps)')
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig('results/comparison_rmse.png')
    plt.show()

    # 4. Bar chart pour la MAPE (Mean Absolute Percentage Error)
    mapes = [r.get('mape', 0.0) * 100 for r in results_list] # Affiché en %
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, mapes, color='orange')
    plt.title('Comparaison de la Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}%", va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig('results/comparison_mape.png')
    plt.show()

    # 5. Bar chart pour le Coefficient de Détermination R²
    r2s = [r.get('r2', 0.0) for r in results_list]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, r2s, color='orchid')
    plt.title('Comparaison du Coefficient de Détermination (R²)')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8) # Ligne de référence à 0
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom' if yval >= 0 else 'top', ha='center')
    plt.tight_layout()
    plt.savefig('results/comparison_r2.png')
    plt.show()

    # 6. Bar chart pour le temps d'exécution (existant)
    plot_execution_times(results_list)
    
    # 7. Dashboard de Synthèse des Métriques (Idéal pour soutenance orale)
    plot_metrics_dashboard(results_list)

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

def plot_monte_carlo(mc_results: dict):
    """
    Génère des boîtes à moustaches (Boxplots) comparatives pour l'analyse Monte Carlo
    sur les métriques MAE, RMSE, MAPE et R² de manière rétro-compatible.
    """
    ensure_results_dir()
    
    # Extraire uniquement les modèles ayant des résultats
    active_models = [name for name, metrics in mc_results.items() if len(metrics['mae']) > 0]
    if not active_models:
        print("Aucun résultat disponible pour tracer l'analyse Monte Carlo.")
        return
        
    mae_data = [mc_results[name]['mae'] for name in active_models]
    rmse_data = [mc_results[name]['rmse'] for name in active_models]
    mape_data = [[v * 100 for v in mc_results[name]['mape']] for name in active_models] # En %
    r2_data = [mc_results[name]['r2'] for name in active_models]
    
    # 1. Boxplot MAE
    plt.figure(figsize=(11, 6))
    plt.boxplot(mae_data)
    plt.title("Distribution de l'Erreur Moyenne Absolue (MAE) - Monte Carlo", fontsize=12, fontweight='bold')
    plt.ylabel("MAE (Mbps)")
    plt.xticks(range(1, len(active_models) + 1), active_models, rotation=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/montecarlo_mae.png', dpi=150)
    plt.show()

    # 2. Boxplot RMSE
    plt.figure(figsize=(11, 6))
    plt.boxplot(rmse_data)
    plt.title("Distribution de la Root Mean Squared Error (RMSE) - Monte Carlo", fontsize=12, fontweight='bold')
    plt.ylabel("RMSE (Mbps)")
    plt.xticks(range(1, len(active_models) + 1), active_models, rotation=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/montecarlo_rmse.png', dpi=150)
    plt.show()

    # 3. Boxplot MAPE
    plt.figure(figsize=(11, 6))
    plt.boxplot(mape_data)
    plt.title("Distribution de la Mean Absolute Percentage Error (MAPE) - Monte Carlo", fontsize=12, fontweight='bold')
    plt.ylabel("MAPE (%)")
    plt.xticks(range(1, len(active_models) + 1), active_models, rotation=25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/montecarlo_mape.png', dpi=150)
    plt.show()

    # 4. Boxplot R²
    plt.figure(figsize=(11, 6))
    plt.boxplot(r2_data)
    plt.title("Distribution du Coefficient de Détermination (R²) - Monte Carlo", fontsize=12, fontweight='bold')
    plt.ylabel("R²")
    plt.xticks(range(1, len(active_models) + 1), active_models, rotation=25)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/montecarlo_r2.png', dpi=150)
    plt.show()
    
    # 5. Dashboard Global Monte Carlo (2x2 Boxplots)
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Tableau de Bord Monte Carlo - Robustesse de la Prédiction Satellite", fontsize=16, fontweight='bold', y=0.98)
    
    # MAE
    axs[0, 0].boxplot(mae_data)
    axs[0, 0].set_title("Distribution de la MAE (Mbps)", fontsize=11, fontweight='bold')
    axs[0, 0].set_ylabel("MAE (Mbps)")
    axs[0, 0].set_xticks(range(1, len(active_models) + 1))
    axs[0, 0].set_xticklabels(active_models, rotation=20)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # RMSE
    axs[0, 1].boxplot(rmse_data)
    axs[0, 1].set_title("Distribution de la RMSE (Mbps)", fontsize=11, fontweight='bold')
    axs[0, 1].set_ylabel("RMSE (Mbps)")
    axs[0, 1].set_xticks(range(1, len(active_models) + 1))
    axs[0, 1].set_xticklabels(active_models, rotation=20)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # MAPE
    axs[1, 0].boxplot(mape_data)
    axs[1, 0].set_title("Distribution de la MAPE (%)", fontsize=11, fontweight='bold')
    axs[1, 0].set_ylabel("MAPE (%)")
    axs[1, 0].set_xticks(range(1, len(active_models) + 1))
    axs[1, 0].set_xticklabels(active_models, rotation=20)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # R2
    axs[1, 1].boxplot(r2_data)
    axs[1, 1].set_title("Distribution du R²", fontsize=11, fontweight='bold')
    axs[1, 1].set_ylabel("R²")
    axs[1, 1].set_xticks(range(1, len(active_models) + 1))
    axs[1, 1].set_xticklabels(active_models, rotation=20)
    axs[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/montecarlo_dashboard.png', dpi=150)
    plt.show()

def plot_metrics_dashboard(results_list: list):
    """
    Génère un tableau de bord 2x2 contenant les 4 métriques de performance
    (MAE, RMSE, MAPE, R²) pour tous les modèles testés.
    """
    ensure_results_dir()
    if not results_list:
        return
        
    names = [r['model_name'] for r in results_list]
    maes = [r.get('mae', 0.0) for r in results_list]
    rmses = [r.get('rmse', 0.0) for r in results_list]
    mapes = [r.get('mape', 0.0) * 100 for r in results_list]
    r2s = [r.get('r2', 0.0) for r in results_list]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Dashboard de Performance des Modèles de Prédiction Satellite', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. MAE
    bars1 = axs[0, 0].bar(names, maes, color='skyblue')
    axs[0, 0].set_title('Erreur Moyenne Absolue (MAE) - Mbps [Plus bas est mieux]', fontsize=11, fontweight='bold')
    axs[0, 0].set_ylabel('MAE (Mbps)')
    axs[0, 0].tick_params(axis='x', rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", va='bottom', ha='center', fontsize=9)
        
    # 2. RMSE
    bars2 = axs[0, 1].bar(names, rmses, color='salmon')
    axs[0, 1].set_title('Root Mean Squared Error (RMSE) - Mbps [Plus bas est mieux]', fontsize=11, fontweight='bold')
    axs[0, 1].set_ylabel('RMSE (Mbps)')
    axs[0, 1].tick_params(axis='x', rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", va='bottom', ha='center', fontsize=9)
        
    # 3. MAPE
    bars3 = axs[1, 0].bar(names, mapes, color='orange')
    axs[1, 0].set_title('Mean Absolute Percentage Error (MAPE) - % [Plus bas est mieux]', fontsize=11, fontweight='bold')
    axs[1, 0].set_ylabel('MAPE (%)')
    axs[1, 0].tick_params(axis='x', rotation=25)
    for bar in bars3:
        yval = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}%", va='bottom', ha='center', fontsize=9)
        
    # 4. R²
    bars4 = axs[1, 1].bar(names, r2s, color='orchid')
    axs[1, 1].set_title('Coefficient de Détermination (R²) [Plus haut est mieux, max=1]', fontsize=11, fontweight='bold')
    axs[1, 1].set_ylabel('R²')
    axs[1, 1].tick_params(axis='x', rotation=25)
    axs[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
    for bar in bars4:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", va='bottom' if yval >= 0 else 'top', ha='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('results/comparison_metrics_dashboard.png', dpi=150)
    plt.show()

