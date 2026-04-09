import pandas as pd
import numpy as np

def generate_fake_traffic_data(
    start_date: str = '2026-03-31 00:00:00', 
    duration_hours: int = 24, 
    link_type: str = 'forward',
    num_beams: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère un faux jeu de données de trafic satellite à la seconde.
    Intègre les comportements de Voie Aller (Forward) vs Voie Retour (Return),
    ainsi que la dimension spatiale (Multifaisceaux).
    
    Args:
        start_date (str): Date et heure de début format 'YYYY-MM-DD HH:MM:SS'
        duration_hours (int): Durée générée en heures.
        link_type (str): 'forward' pour Voie Aller, 'return' pour Voie Retour.
        num_beams (int): Nombre de faisceaux à simuler.
        seed (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        pd.DataFrame: DataFrame avec un index temporel et 'num_beams' colonnes de trafic.
    """
    np.random.seed(seed)
    
    # Fréquence 'S' pour secondes
    periods = duration_hours * 3600
    dates = pd.date_range(start=start_date, periods=periods, freq='s')
    
    time_seconds = np.arange(periods)
    daily_cycle = np.sin(2 * np.pi * time_seconds / (24 * 3600))
    
    data_dict = {}
    
    for beam in range(1, num_beams + 1):
        if link_type == 'forward':
            # Voie Aller : trafic descendant massif et lisse
            base_traffic = 100.0 + np.random.uniform(-10, 10)
            amplitude = 40.0 + np.random.uniform(-5, 5)
            noise = np.random.normal(0, 1.0, periods)
            traffic_volume = base_traffic + amplitude * daily_cycle + noise
        else:
            # Voie Retour : trafic montant, plus faible mais fragmenté/bursty
            base_traffic = 20.0 + np.random.uniform(-5, 5)
            amplitude = 10.0 + np.random.uniform(-2, 2)
            noise = np.random.normal(0, 3.0, periods)
            bursts = np.random.poisson(0.1, periods) * np.random.uniform(10, 30, periods)
            traffic_volume = base_traffic + amplitude * daily_cycle + noise + bursts
            
        # On évite les valeurs négatives
        data_dict[f'beam_{beam}'] = np.maximum(traffic_volume, 0)
        
    df = pd.DataFrame(data_dict, index=dates)
    
    return df

if __name__ == '__main__':
    print("Génération de test de 1 heure de trafic (Voie Aller, 3 Faisceaux)...")
    df = generate_fake_traffic_data(duration_hours=1, link_type='forward', num_beams=3)
    print(df.head())
    print(f"Total des lignes : {len(df)}")
