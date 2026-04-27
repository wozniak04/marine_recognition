import folium
from folium.plugins import MarkerCluster
import csv

def stworz_mape(plik_wejsciowy, plik_wyjsciowy):
    szerokosci = []
    dlugosci = []
    punkty = []

    try:
        with open(plik_wejsciowy, mode='r', encoding='utf-8') as plik:
            czytnik = csv.reader(plik)
            next(czytnik, None) # Pomijamy nagłówek
            
            for wiersz in czytnik:
                # Sprawdzamy, czy wiersz ma kolumny 0, 1 i 2
                if len(wiersz) >= 3:
                    try:
                        opis = wiersz[0].strip()
                        lat = float(wiersz[1].strip())
                        lon = float(wiersz[2].strip())
                        
                        punkty.append((lat, lon, opis))
                        szerokosci.append(lat)
                        dlugosci.append(lon)
                    except ValueError:
                        continue # Pomija błędne dane w wierszu
                    
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{plik_wejsciowy}'.")
        return

    if not punkty:
        print("Nie znaleziono żadnych prawidłowych współrzędnych.")
        return

    srodek_lat = sum(szerokosci) / len(szerokosci)
    srodek_lon = sum(dlugosci) / len(dlugosci)

    # Optymalizacja 1: prefer_canvas=True sprawia, że mapa działa płynniej
    mapa = folium.Map(
        location=[srodek_lat, srodek_lon], 
        zoom_start=5, 
        prefer_canvas=True
    )

    # Optymalizacja 2: MarkerCluster - grupuje punkty, by nie lagowały przy oddaleniu
    marker_cluster = MarkerCluster().add_to(mapa)

    for lat, lon, opis in punkty:
        folium.Marker(
            location=[lat, lon],
            popup=opis,
            tooltip="Kliknij po szczegóły"
        ).add_to(marker_cluster)

    mapa.save(plik_wyjsciowy)
    print(f"Sukces! Mapa została zoptymalizowana i zapisana jako: {plik_wyjsciowy}")

# Uruchomienie z Twoją ścieżką
stworz_mape('./Example_data_project_description_v3/EU_Port_codes.csv', 'moja_mapa.html')