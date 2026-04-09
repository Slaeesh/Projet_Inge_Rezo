import pandas as pd
import numpy as np

def generate_fake_traffic_data(start_date='2026-03-31 00:00:00', duration_hours=24, seed=42):
    """
    Génère un faux jeu de données de trafic satellite à la seconde.
    Inclut une tendance quotidienne (sinusoïdale) et du bruit aléatoire.
    
    Args:
        start_date (str): Date et heure de début format 'YYYY-MM-DD HH:MM:SS'
        duration_hours (int): Durée générée en heures.
        seed (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        pd.DataFrame: DataFrame avec un index temporel et le volume de trafic.
    """
    np.random.seed(seed)
    
    # Fréquence 'S' pour secondes
    periods = duration_hours * 3600
    dates = pd.date_range(start=start_date, periods=periods, freq='s')
    
    # Création d'un signal "trafic" avec un cycle journalier
    # On ajoute une composante sinusoïdale (période 24h) et un bruit blanc.
    time_seconds = np.arange(periods)
    daily_cycle = np.sin(2 * np.pi * time_seconds / (24 * 3600))
    noise = np.random.normal(0, 0.2, periods)
    
    # Formule arbitraire pour le volume de données (ex: Mbps)
    # L'objectif est d'avoir des valeurs positives simulant une charge réseau
    base_traffic = 50.0  # Mbps
    amplitude = 30.0
    traffic_volume = base_traffic + amplitude * daily_cycle + (noise * 10)
    
    # On évite les valeurs négatives
    traffic_volume = np.maximum(traffic_volume, 0)
    
    df = pd.DataFrame({
        'traffic_mbps': traffic_volume
    }, index=dates)
    
    return df

if __name__ == '__main__':
    print("Génération de test de 1 heure de trafic...")
    df = generate_fake_traffic_data(duration_hours=1)
    print(df.head())
    print(f"Total des lignes : {len(df)}")
