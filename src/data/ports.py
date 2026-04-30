import pandas as pd
from scipy.spatial import cKDTree

def load_ports_data(file_path):
    return pd.read_csv(file_path)

ports=load_ports_data('training_data/EU_Port_Codes.csv')
tree = cKDTree(ports[['LAT', 'LON']].values)

def check_if_in_port(lat, lon, radius_meters=2000):
    # Przeliczamy metry na stopnie (uproszczenie: 111 000m ok. 1 stopień)
    radius_degrees = radius_meters / 111000
    # Szukamy wszystkich portów w tym promieniu
    indices = tree.query_ball_point([lat, lon], r=radius_degrees)
    
    if not indices:
        return 0  
    
    # Jeśli chcesz tylko jeden (pierwszy znaleziony) port:
    return 1

# Example usage:
#print(check_if_in_port(17.9000001,-62.85000001))