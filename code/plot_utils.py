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

def plot_exhaustive_benchmark(results_list: list):
    """
    Génère le tableau de bord de synthèse du benchmark exhaustif (Mode 6) :
    1. Heatmap du score R² moyen.
    2. Heatmap de la MAPE moyenne (%).
    3. Barplot du nombre de configurations remportées.
    4. Barplot du temps moyen d'exécution par run.
    """
    ensure_results_dir()
    if not results_list:
        print("Aucun résultat disponible pour tracer le benchmark exhaustif.")
        return
        
    import pandas as pd
    df = pd.DataFrame(results_list)
    
    # 1. Calcul du nombre de configurations remportées (Win Count)
    win_counts = {m: 0 for m in df['model_name'].unique()}
    group_cols = ['link_type', 'freq', 'beam_name', 'horizon']
    has_scenario = 'scenario' in df.columns
    if has_scenario:
        group_cols = ['scenario'] + group_cols
        
    groups = df.groupby(group_cols)
    total_victories = len(groups)
    
    for g_keys, g_df in groups:
        best_row = g_df.loc[g_df['mae'].idxmin()]
        win_counts[best_row['model_name']] += 1
        
    # 2. Préparation des données pour les Heatmaps
    df['link_short'] = df['link_type'].apply(lambda x: 'TX' if x == 'forward' else 'RX')
    df['config_group'] = df['link_short'] + '_' + df['freq'] + '_H' + df['horizon'].astype(str)
    
    config_groups = sorted(df['config_group'].unique())
    models = sorted(df['model_name'].unique())
    
    r2_heatmap_data = np.zeros((len(models), len(config_groups)))
    mape_heatmap_data = np.zeros((len(models), len(config_groups)))
    
    for i, m in enumerate(models):
        for j, cg in enumerate(config_groups):
            subset = df[(df['model_name'] == m) & (df['config_group'] == cg)]
            if not subset.empty:
                r2_heatmap_data[i, j] = subset['r2'].mean()
                mape_heatmap_data[i, j] = subset['mape'].mean() * 100
            else:
                r2_heatmap_data[i, j] = np.nan
                mape_heatmap_data[i, j] = np.nan
                
    # 3. Calcul des temps d'exécution moyens
    mean_times = df.groupby('model_name')['execution_time'].mean().to_dict()
    
    # Tracé (Grille 2x2)
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    
    # --- SUBPLOT 1: Heatmap R² (Top-Left) ---
    im1 = axs[0, 0].imshow(r2_heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)
    axs[0, 0].set_title("1. Heatmap du Score R² Moyen par Configuration (Moyenne Beams/Scénarios)", fontsize=13, fontweight='bold', pad=12)
    axs[0, 0].set_yticks(range(len(models)))
    axs[0, 0].set_yticklabels(models, fontsize=10, fontweight='bold')
    axs[0, 0].set_xticks(range(len(config_groups)))
    axs[0, 0].set_xticklabels(config_groups, rotation=35, ha='right', fontsize=9)
    axs[0, 0].set_xlabel("Configurations (Voie_Fréquence_Horizon)", fontsize=11, labelpad=8)
    
    for i in range(len(models)):
        for j in range(len(config_groups)):
            val = r2_heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "black" if val > 0.4 else "white"
                axs[0, 0].text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
    fig.colorbar(im1, ax=axs[0, 0], label="Coefficient de Détermination R²")
    
    # --- SUBPLOT 2: Heatmap MAPE (Top-Right) ---
    im2 = axs[0, 1].imshow(mape_heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=30.0)
    axs[0, 1].set_title("2. Heatmap de la MAPE Moyenne (%) par Configuration (Moyenne Beams/Scénarios)", fontsize=13, fontweight='bold', pad=12)
    axs[0, 1].set_yticks(range(len(models)))
    axs[0, 1].set_yticklabels(models, fontsize=10, fontweight='bold')
    axs[0, 1].set_xticks(range(len(config_groups)))
    axs[0, 1].set_xticklabels(config_groups, rotation=35, ha='right', fontsize=9)
    axs[0, 1].set_xlabel("Configurations (Voie_Fréquence_Horizon)", fontsize=11, labelpad=8)
    
    for i in range(len(models)):
        for j in range(len(config_groups)):
            val = mape_heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "black" if val < 15.0 else "white"
                axs[0, 1].text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
    fig.colorbar(im2, ax=axs[0, 1], label="Erreur Absolue Moyenne en Pourcentage (MAPE %)")
    
    # --- SUBPLOT 3: Win Count Barplot (Bottom-Left) ---
    model_names_win = list(win_counts.keys())
    counts_win = list(win_counts.values())
    sorted_win_idx = np.argsort(counts_win)[::-1]
    model_names_win = [model_names_win[idx] for idx in sorted_win_idx]
    counts_win = [counts_win[idx] for idx in sorted_win_idx]
    
    bars_win = axs[1, 0].bar(model_names_win, counts_win, color='teal', edgecolor='black', alpha=0.85)
    axs[1, 0].set_title("3. Nombre de Configurations Remportées (Plus petite MAE)", fontsize=13, fontweight='bold', pad=12)
    axs[1, 0].set_ylabel(f"Nombre de victoires (Total {total_victories})", fontsize=11)
    axs[1, 0].set_xticks(range(len(model_names_win)))
    axs[1, 0].set_xticklabels(model_names_win, rotation=25, ha='right', fontsize=10, fontweight='bold')
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars_win:
        yval = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width()/2, yval + (total_victories * 0.01), str(int(yval)), va='bottom', ha='center', fontsize=11, fontweight='bold')
        
    # --- SUBPLOT 4: Execution Times Comparison (Bottom-Right) ---
    model_names_time = list(mean_times.keys())
    times = list(mean_times.values())
    sorted_time_idx = np.argsort(times) # Tri croissant
    model_names_time = [model_names_time[idx] for idx in sorted_time_idx]
    times = [times[idx] for idx in sorted_time_idx]
    
    bars_time = axs[1, 1].bar(model_names_time, times, color='coral', edgecolor='black', alpha=0.85)
    axs[1, 1].set_title("4. Temps d'Exécution Moyen par Run (Entraînement + Inférence)", fontsize=13, fontweight='bold', pad=12)
    axs[1, 1].set_ylabel("Temps moyen (secondes)", fontsize=11)
    axs[1, 1].set_xticks(range(len(model_names_time)))
    axs[1, 1].set_xticklabels(model_names_time, rotation=25, ha='right', fontsize=10, fontweight='bold')
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars_time:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval + (max(times) * 0.01), f"{yval:.4f}s", va='bottom', ha='center', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('results/exhaustive_benchmark.png', dpi=150)
    axs[0, 0].figure.canvas.draw()
    plt.close(fig)
    print("Graphique de synthèse exhaustif sauvegardé dans 'results/exhaustive_benchmark.png'.")


