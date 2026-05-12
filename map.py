import streamlit as st
import pandas as pd
import plotly.express as px

# Ustawienia strony
st.set_page_config(page_title="Wizualizacja Tras", layout="wide")
st.title("Wizualizacja Statków na Mapie")

# Wczytywanie pliku CSV
uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")
to_highlight = "signaldate"
if uploaded_file is not None:
    # Wczytanie całego pliku do pamięci, aby poznać jego rozmiar
    if "data" in uploaded_file.name:
        to_highlight = "signaldate"
    df = pd.read_csv(uploaded_file)
    max_index = len(df) - 1
    
    st.write(f"Wczytano plik. Liczba wierszy: {len(df)}")
    
    # Wybór zakresu indeksów do wyświetlenia
    col1, col2 = st.columns(2)
    with col1:
        start_index = st.number_input("Początkowy indeks", min_value=0, max_value=max_index, value=0)
    with col2:
        end_index = st.number_input("Końcowy indeks", min_value=0, max_value=max_index, value=min(1000, max_index))
        
    if start_index <= end_index:
        # Filtrowanie dataframe na podstawie wybranych indeksów
        df_filtered = df.iloc[start_index:end_index + 1].copy()
        
        # Funkcja do określania pojedynczego statusu statku
        def get_status(row):
            if row.get('is_outlier', 0) == 1: return 'Outlier GPS'
            if row.get('In Port', 0) == 1: return 'In Port'
            if row.get('At Sea Voyage', 0) == 1: return 'At Sea Voyage'
            if row.get('At Sea Anchor', 0) == 1: return 'At Sea Anchor'
            if row.get('is_drift', 0) == 1: return 'At Sea Adrift GPS'
            return 'Nieznany'
            
        # Tworzymy nową kolumnę "Status" dla łatwego kolorowania
        df_filtered['Status'] = df_filtered.apply(get_status, axis=1)
        
        # Definiujemy wybrane kolory dla danego statusu
        color_map = {
            'Outlier GPS': 'red',
            'In Port': 'purple',
            'At Sea Voyage': 'blue',
            'At Sea Anchor': 'orange',
            'At Sea Adrift GPS': 'green',
            'Nieznany': 'blue'
        }
        
        # Generowanie mapy
        fig = px.scatter_mapbox(
            df_filtered, 
            lat="LAT", 
            lon="LON", 
            color="Status",               # Na tej podstawie dobierane są kolory i tworzona legenda
            color_discrete_map=color_map, # Przypisanie konkretnych kolorów
            hover_name=to_highlight,      # Wyświetla signaldate na samym szczycie dymku
            hover_data={"LAT": False, "LON": False, "Status": False}, # Ukrywa zbędne info z dymka
            zoom=10, 
            height=700
        )
        
        # Wygląd mapy (darmowa otwarta mapa, nie wymaga tokenu)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        # Renderowanie na stronie
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Początkowy indeks nie może być większy niż końcowy.")
