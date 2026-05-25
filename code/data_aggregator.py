import pandas as pd

def aggregate_data(df, freq):
    """
    Agrège les données de trafic selon la fréquence spécifiée.
    
    Args:
        df (pd.DataFrame): Le DataFrame d'origine (indexé par paquets/secondes).
        freq (str): Fréquence d'agrégation, ex: '1s', '1min' (pandas '1T' / '1min'), '10min' (pandas '10T' / '10min').
        
    Returns:
        pd.DataFrame: Le DataFrame agrégé (moyenne du trafic sur la période).
    """
    # Mapping simple pour utiliser les alias Pandas standards si besoin
    freq_map = {
        '1s': 's',
        '1min': 'min',
        '10min': '10min'
    }
    
    target_freq = freq_map.get(freq, freq)
    
    # On agrège en faisant la moyenne du trafic sur l'intervalle
    # On pourrait aussi faire la somme selon la définition de la métrique.
    df_agg = df.resample(target_freq).mean()
    
    # Remplir les éventuelles valeurs manquantes générées par le rééchantillonnage
    df_agg = df_agg.ffill()
    df_agg.dropna(inplace=True)
    
    return df_agg

if __name__ == '__main__':
    from data_generator import generate_fake_traffic_data
    
    # Test
    df = generate_fake_traffic_data(duration_hours=1)
    print(f"Original shape (1s): {df.shape}")
    
    df_1m = aggregate_data(df, '1min')
    print(f"Aggregated shape (1min): {df_1m.shape}")
    print(df_1m.head())
    
    df_10m = aggregate_data(df, '10min')
    print(f"Aggregated shape (10min): {df_10m.shape}")
    print(df_10m.head())
