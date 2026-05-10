# ==============================================================================
# CELL 1: Importy i funkcje pomocnicze
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from geopy.distance import geodesic

warnings.filterwarnings('ignore')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'signaldate' in df.columns:
        df['signaldate'] = pd.to_datetime(df['signaldate'])
    return df


# ==============================================================================
# CELL 2: Wykrycie outlierów 
# ==============================================================================
def calculate_bearing(lat1, lon1, lat2, lon2):
    try:
        p_lat, p_lon, c_lat, c_lon = np.radians([float(lat1), float(lon1), float(lat2), float(lon2)])
        d_lon = c_lon - p_lon
        y = np.sin(d_lon) * np.cos(c_lat)
        x = np.cos(p_lat) * np.sin(c_lat) - np.sin(p_lat) * np.cos(c_lat) * np.cos(d_lon)
        return (np.degrees(np.arctan2(y, x)) + 360) % 360
    except Exception:
        return np.nan

def detect_and_handle_outliers(df):
    HIGHEST_POSSIBLE_SOG = 60.0
    HIGHEST_POSSIBLE_ACCELERATION = 0.2
    HIGHEST_POSSIBLE_TURN = 1.0
    df["LAT"] = pd.to_numeric(df["LAT"], errors='coerce')
    df["LON"] = pd.to_numeric(df["LON"], errors='coerce')
    df["signaldate"] = pd.to_datetime(df["signaldate"])

    # Sortowanie chronologiczne
    df = df.sort_values(by="signaldate").reset_index(drop=True)

    # Listy na wyniki (muszą mieć tę samą długość co df)
    out_sog, out_cog, out_accel, out_rot, is_outlier = [], [], [], [], []

    last_valid_row = None
    last_valid_sog = 0.0
    last_valid_cog = np.nan

    for index, row in df.iterrows():
        if index % 5000 == 0:
            print(f"Przetwarzanie: {index}")

        # --- KROK 1: SPRAWDZENIE CZY DANE GPS W OGÓLE ISTNIEJĄ ---
        if pd.isna(row["LAT"]) or pd.isna(row["LON"]):
            out_sog.append(np.nan)
            out_cog.append(np.nan)
            out_accel.append(np.nan)
            out_rot.append(np.nan)
            is_outlier.append(1) # Brak współrzędnych = Outlier
            continue

        # --- KROK 2: JEŚLI TO PIERWSZY PRAWIDŁOWY PUNKT ---
        if last_valid_row is None:
            out_sog.append(0.0)
            out_cog.append(np.nan)
            out_accel.append(0.0)
            out_rot.append(0.0)
            is_outlier.append(0)
            last_valid_row = row
            last_valid_sog = 0.0
            continue

        # --- KROK 3: OBLICZENIA WZGLĘDEM OSTATNIEGO DOBREGO PUNKTU ---
        time_diff = (row["signaldate"] - last_valid_row["signaldate"]).total_seconds()

        if time_diff <= 0:
            # Duplikat czasu lub błąd w danych - oznaczamy i nie przesuwamy punktu odniesienia
            out_sog.append(last_valid_sog)
            out_cog.append(last_valid_cog)
            out_accel.append(0.0)
            out_rot.append(0.0)
            is_outlier.append(1)
            continue

        # Dystans
        try:
            prev_pos = (float(last_valid_row["LAT"]), float(last_valid_row["LON"]))
            curr_pos = (float(row["LAT"]), float(row["LON"]))
            dist_meters = geodesic(prev_pos, curr_pos).meters
        except:
            dist_meters = 0.0

        # SOG i COG
        current_sog = (dist_meters / time_diff) * 1.94384
        current_cog = calculate_bearing(prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])

        # Akceleracja (m/s^2)
        curr_sog_ms = current_sog * 0.51444
        prev_sog_ms = last_valid_sog * 0.51444
        current_accel = (curr_sog_ms - prev_sog_ms) / time_diff

        # ROT
        p_cog = last_valid_cog if not pd.isna(last_valid_cog) else current_cog
        delta_cog = (current_cog - p_cog + 180) % 360 - 180
        current_rot = (delta_cog / time_diff) * 60

        # --- KROK 4: WERYFIKACJA CZY PUNKT NIE JEST SKOKIEM ---
        outlier_flag = 0
        if dist_meters == 0:
            outlier_flag = 1
        elif current_sog > HIGHEST_POSSIBLE_SOG:
            outlier_flag = 1
        elif abs(current_accel) > HIGHEST_POSSIBLE_ACCELERATION:
            outlier_flag = 1
        # Filtr skrętu (opcjonalny, zależy od typu jednostki)
        elif current_sog > 1.0 and (current_sog * abs(current_rot) / 1100) > HIGHEST_POSSIBLE_TURN:
            outlier_flag = 1
        

        # --- KROK 5: ZAPIS I AKTUALIZACJA REFERENCJI ---
        out_sog.append(current_sog)
        out_cog.append(current_cog)
        out_accel.append(current_accel)
        out_rot.append(current_rot)
        is_outlier.append(outlier_flag)

        if outlier_flag == 0:
            # TYLKO jeśli punkt jest poprawny, staje się nową bazą do następnych obliczeń
            last_valid_row = row
            last_valid_sog = current_sog
            last_valid_cog = current_cog


    df["Outlier"] = is_outlier

    outliers_count = sum(is_outlier)
    print(f"Zakończono wykrywanie outlierów. Znaleziono {outliers_count} punktów odstających spośród {len(df)} rekordów.")
    
    # Filtrowanie i usunięcie outlierów z datasetu
    df_clean = df[df["Outlier"] == 0].reset_index(drop=True)
    print(f"Dane po oczyszczeniu: {len(df_clean)} rekordów.")
    return df_clean


# ==============================================================================
# CELL 3: Preprocessing oraz BallTree do portów
# ==============================================================================
class DataPreprocessor:
    """Czyszczenie danych GPS i obliczanie kinematyki"""
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_proc = df.copy()
        
        df_proc = df_proc.dropna(subset=["LAT", "LON"]).reset_index(drop=True)
        df_proc["signaldate"] = pd.to_datetime(df_proc["signaldate"])
        df_proc = df_proc.drop_duplicates(subset=["signaldate"])
        df_proc = df_proc.sort_values("signaldate").reset_index(drop=True)

        dt = np.empty(len(df_proc))
        dt[0] = np.nan
        dt[1:] = df_proc["signaldate"].diff().dt.total_seconds().values[1:]
        df_proc["dt"] = dt
        
        is_gap = np.zeros(len(df_proc), dtype=bool)
        is_gap[0] = True
        is_gap[1:] = dt[1:] > 300
        df_proc["is_gap"] = is_gap.astype(int)
        
        lat = df_proc["LAT"].values
        lon = df_proc["LON"].values
        dist = np.empty(len(df_proc))
        dist[0] = np.nan
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dist[is_gap] = np.nan
        df_proc["distance_m"] = dist
        
        with np.errstate(divide="ignore", invalid="ignore"):
            sog_ms = dist / dt
        sog_ms[is_gap] = np.nan
        df_proc["sog_ms"] = sog_ms
        df_proc["sog_knots"] = sog_ms * 1.943844
        
        raw_cog = np.empty(len(df_proc))
        raw_cog[0] = np.nan
        raw_cog[1:] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
        
        dcog = np.empty(len(df_proc))
        dcog[:2] = np.nan
        dcog[2:] = (raw_cog[2:] - raw_cog[1:-1] + 180) % 360 - 180
        dcog[is_gap] = np.nan
        df_proc["dcog_deg"] = dcog
        
        return df_proc

def assign_in_port(df: pd.DataFrame, ports_csv_path: str, radius_m: float = 3000.0) -> pd.DataFrame:
    """Sprawdza, czy punkty znajdują się w promieniu portu przy pomocy BallTree."""
    df_out = df.copy()
    
    # Jeśli ma kolumnę z etykietą, to nie musimy szukać na nowo (Trening)
    if "In Port" in df_out.columns:
        return df_out
        
    try:
        ports_df = pd.read_csv(ports_csv_path)
    except Exception as e:
        print(f"Błąd wczytywania bazy portów: {e}")
        df_out["In Port"] = 0
        return df_out
        
    EARTH_RADIUS_M = 6_371_000
    coords_rad_ports = np.radians(ports_df[["LAT", "LON"]].values)
    tree = BallTree(coords_rad_ports, metric="haversine")
    
    coords_rad_query = np.radians(df_out[["LAT", "LON"]].values)
    dist_rad, _ = tree.query(coords_rad_query, k=1)
    dist_m = dist_rad[:, 0] * EARTH_RADIUS_M
    
    df_out["In Port"] = (dist_m <= radius_m).astype(int)
    return df_out

def build_features_and_route(df: pd.DataFrame, window_size: int = 5):
    """Preprocesing pod model budujący sekwencje czasowe."""
    df = df.copy()
    sog = df["sog_knots"].fillna(0).values
    dcog = df["dcog_deg"].fillna(0).values
    lats = df["LAT"].values
    lons = df["LON"].values
    timestamps = df["signaldate"].values
    
    is_port = df["In Port"].values == 1 if "In Port" in df.columns else np.zeros(len(df), dtype=bool)
    
    ptp_dists = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
    ptp_dists = np.insert(ptp_dists, 0, 0.0)

    X_sequences, is_port_sequence, indices = [], [], []
    
    for i in range(len(df) - window_size + 1):
        w_lats = lats[i : i + window_size]
        w_lons = lons[i : i + window_size]
        
        dist_m = haversine(np.array([w_lats[0]]), np.array([w_lons[0]]), np.array([w_lats[-1]]), np.array([w_lons[-1]]))[0]
        time_delta_seconds = (timestamps[i + window_size - 1] - timestamps[i]) / np.timedelta64(1, 's')
        window_speed_ms = dist_m / time_delta_seconds if time_delta_seconds > 0 else 0.0

        c_lat, c_lon = w_lats.mean(), w_lons.mean()
        dists_to_center = haversine(w_lats, w_lons, np.full(window_size, c_lat), np.full(window_size, c_lon))
        spread_m = dists_to_center.max()
        
        path_dist = np.sum(ptp_dists[i+1 : i+window_size])
        efficiency = dist_m / path_dist if path_dist > 0 else 0.0
        window_turn_sum = np.sum(np.abs(dcog[i : i + window_size]))
        shape_index = dist_m / (spread_m + 1.0)

        seq = []
        for j in range(window_size):
            seq.append([sog[i+j], dcog[i+j], window_speed_ms, spread_m, dist_m, efficiency, window_turn_sum, shape_index])
            
        X_sequences.append(seq)
        
        center_idx = i + window_size // 2
        is_port_sequence.append(is_port[center_idx])
        indices.append(center_idx)
        
    return np.array(X_sequences), np.array(is_port_sequence), np.array(indices)


# ==============================================================================
# CELL 4: Definicje modeli
# ==============================================================================
class SimpleSeaLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out_mean = torch.mean(out, dim=1)
        return self.fc(out_mean)

class SimpleSeaModelWrapper:
    def __init__(self, epochs=20, lr=0.001, class_weights=None):
        self.epochs = epochs
        self.lr = lr
        self.model = SimpleSeaLSTM()
        self.history = {'train_loss': []}
        
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        if len(X_seq) == 0: return
        self.model.train()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_x), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            self.history['train_loss'].append(avg_loss)
            self.scheduler.step(avg_loss)
            
    def plot_learning_curve(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.history['train_loss']) + 1), self.history['train_loss'], marker='o', label='Train Loss')
        plt.title('Krzywa uczenia (Loss per Epoch)')
        plt.xlabel('Epoka')
        plt.ylabel('Strata (CrossEntropy Loss)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def predict(self, X_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(X_seq) == 0: return np.array([]), np.array([])
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            probs = torch.softmax(self.model(X_tensor), dim=1)
            confs, preds = torch.max(probs, dim=1)
        return preds.numpy(), confs.numpy() * 100.0

class PortManeuverNN:
    def __init__(self): pass
    def fit(self, X, y): pass
    def predict(self, X) -> np.ndarray: return np.zeros(len(X), dtype=int)


# ==============================================================================
# CELL 5: Trenowanie modelu (Wymaga istniejących etykiet i 'In Port')
# ==============================================================================
def train_hierarchical_model(train_df: pd.DataFrame, epochs: int = 25, window_size: int = 5) -> dict:
    print("Budowanie cech i okien dla treningu...")
    X_seq, is_port_seq, idxs = build_features_and_route(train_df, window_size=window_size)
    labels = train_df["operation_id"].values[idxs]
    
    sea_mask = ~is_port_seq
    X_sea = X_seq[sea_mask]
    y_sea = (labels[sea_mask] - 2).astype(int) 
    
    counts = np.bincount(y_sea, minlength=3)
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.power(counts, 0.25)
    weights = (weights / np.sum(weights)) * 3.0
    
    sea_model = SimpleSeaModelWrapper(epochs=epochs, class_weights=weights)
    scaler = StandardScaler()
    if len(X_sea) > 0:
        N, W, F = X_sea.shape
        X_sea_scaled = scaler.fit_transform(X_sea.reshape(-1, F)).reshape(N, W, F)
        print(f"Trenowanie modelu morskiego na {len(X_sea)} oknach...")
        sea_model.fit(X_sea_scaled, y_sea)
        
    port_model = PortManeuverNN()
    if is_port_seq.any():
        print(f"Trenowanie modelu portowego na {is_port_seq.sum()} oknach...")
        port_model.fit(X_seq[is_port_seq], np.zeros(is_port_seq.sum(), dtype=int))
        
    print("Trening zakończony pomyślnie.\n")
    return {"sea_model": sea_model, "port_model": port_model, "scaler": scaler}


# ==============================================================================
# CELL 6: Ewaluacja i predykcja (Symulacja na danych surowych)
# ==============================================================================
def predict_model(trained_dict: dict, df: pd.DataFrame, ports_csv_path: str, window_size: int = 5) -> pd.DataFrame:
    sea_model = trained_dict["sea_model"]
    port_model = trained_dict["port_model"]
    scaler = trained_dict.get("scaler")
    
    # Przed predykcją używamy BallTree do znalezienia portów
    df_with_port = assign_in_port(df, ports_csv_path, radius_m=3000.0)
    X_seq, is_port_seq, idxs = build_features_and_route(df_with_port, window_size=window_size)
    
    result = df_with_port.copy()
    if "predicted_state" not in result.columns: result["predicted_state"] = "adrift"
    if "confidence" not in result.columns: result["confidence"] = 50.0
    
    if is_port_seq.any():
        result.iloc[idxs[is_port_seq], result.columns.get_loc("predicted_state")] = "port_stay"
        result.iloc[idxs[is_port_seq], result.columns.get_loc("confidence")] = 95.0
        
    if (~is_port_seq).any():
        X_sea = X_seq[~is_port_seq]
        if scaler is not None and len(X_sea) > 0:
            N, W, F = X_sea.shape
            X_sea = scaler.transform(X_sea.reshape(-1, F)).reshape(N, W, F)
            
        sea_preds, sea_confs = sea_model.predict(X_sea)
        SEA_STATE_MAP = {0: "anchor", 1: "adrift", 2: "voyage"}
        pred_strings = [SEA_STATE_MAP.get(p, "adrift") for p in sea_preds]
        
        result.iloc[idxs[~is_port_seq], result.columns.get_loc("predicted_state")] = pred_strings
        result.iloc[idxs[~is_port_seq], result.columns.get_loc("confidence")] = sea_confs
        
    result["predicted_state"] = result["predicted_state"].ffill().bfill()
    result["confidence"] = result["confidence"].ffill().bfill()
    return result

def evaluate_pipeline(trained_dict: dict, df_true: pd.DataFrame, ports_csv_path: str, dataset_name: str = ""):
    """Odpala predykcję na danych (wymuszając użycie BallTree) i oblicza Accuracy."""
    # Ukrywamy 'In Port' dla predykcji, żeby pipeline testował działanie BallTree
    df_eval = df_true.copy()
    if "In Port" in df_eval.columns:
        df_eval = df_eval.drop(columns=["In Port"])
        
    preds_df = predict_model(trained_dict, df_eval, ports_csv_path, window_size=5)
    
    ID_TO_STATE = {1: "port_stay", 2: "anchor", 3: "adrift", 4: "voyage"}
    y_true_str = df_true["operation_id"].map(ID_TO_STATE)
    y_pred_str = preds_df["predicted_state"]
    
    valid_mask = y_pred_str.notna() & y_true_str.notna()
    y_true_clean = y_true_str[valid_mask]
    y_pred_clean = y_pred_str[valid_mask]
    
    acc = accuracy_score(y_true_clean, y_pred_clean)
    print(f"[{dataset_name}] Accuracy: {acc * 100:.2f}%")
    
    return acc, preds_df, y_true_clean, y_pred_clean


# ==============================================================================
# CELL 7: Główny skrypt uruchomieniowy
# ==============================================================================


    # --- PODAJ SWOJE ŚCIEŻKI TUTAJ ---
TRAIN_CSV = r"C:\Users\TMMsi\Desktop\marine_recognition\data\raw\classified\Ship_Operation_example_dataset_classified.csv"
PORTS_CSV = r"C:\Users\TMMsi\Desktop\marine_recognition\data\ports\EU_Port_Codes.csv"
PREDICT_CSV = r"C:\Users\TMMsi\Desktop\marine_recognition\data\raw\unclassified\Ship_Operation_example_dataset_unclassified_source_GPS_1.csv"
    
# 1. Wczytanie i globalny preprocesing
print("Wczytywanie i preprocesing danych...")
df_all = load_data(TRAIN_CSV)

df_all = detect_and_handle_outliers(df_all)


preprocessor = DataPreprocessor()
df_all_proc = preprocessor.process(df_all)


    # 2. Podział na zbiory (Train: 70%, Val: 15%, Test: 15%)
n = len(df_all_proc)
train_idx = int(n * 0.70)
val_idx = int(n * 0.85)

df_train = df_all_proc.iloc[:train_idx].copy()
df_val = df_all_proc.iloc[train_idx:val_idx].copy()
df_test = df_all_proc.iloc[val_idx:].copy()
print(f"Rozmiary zbiorów (wiersze) - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}\n")


    # 3. Trenowanie modelu
models = train_hierarchical_model(df_train, epochs=15, window_size=5)

    # 4. Rysowanie krzywej uczenia z loss-em
    # Odkomentuj to w notatniku, żeby narysował się wykres!
models["sea_model"].plot_learning_curve()

    # 5. Ewaluacja - Pokazuje metryki Accuracy
print("\n=== WYNIKI EWALUACJI ===")
_ = evaluate_pipeline(models, df_train, PORTS_CSV, dataset_name="Train")
_, preds_val_df, y_true_val, y_pred_val = evaluate_pipeline(models, df_val, PORTS_CSV, dataset_name="Validation")
_, preds_test_df, y_true_test, y_pred_test = evaluate_pipeline(models, df_test, PORTS_CSV, dataset_name="Test")
    # 6. Rysowanie Confusion Matrix na zbiorze Walidacyjnym
labels = ["port_stay", "anchor", "adrift", "voyage"]
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)

    
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Ilość okien'})
plt.title('Macierz pomyłek (Confusion Matrix) - Zbiór Walidacyjny')
plt.ylabel('Rzeczywista klasa (True)')
plt.xlabel('Przewidziana klasa (Predicted)')
plt.show()

    # 7. Ostateczna predykcja na nowych danych i zapis
    # print("\n=== PREDYKCJA NA NOWYCH DANYCH ===")
    # final_predictions = predict_model(models, df_predict_proc, PORTS_CSV, window_size=5)
    # output_path = r"C:\Users\TMMsi\Desktop\marine_recognition\data\preprocessed\output.csv"
    
    # import os
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # final_predictions.to_csv(output_path, index=False)
    # print(f"Predykcje zapisano pomyślnie do pliku: {output_path}")