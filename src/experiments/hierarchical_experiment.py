from typing import Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.simple_sea_lstm import SimpleSeaModelWrapper
from src.models.port_maneuver_nn import PortManeuverNN
from src.utils.geo import haversine
from src.utils.config import OPERATION_ID_TO_STATE

@register_experiment("hierarchical_lstm")
class HierarchicalExperiment(BaseExperiment):

    @classmethod
    def name(cls) -> str:
        return "hierarchical_lstm"
        
    def _build_features_and_route(self, df: pd.DataFrame, window_size: int = 5):
        """
        Preprocesing: oblicza zadaną prędkość, kąt oraz dystans od 1 do ostatniego punktu okna.
        Dodatkowo wyodrębnia informacje czy środek okna wypadł w porcie.
        """
        df = df.copy()
        
        # BaseExperiment za pomocą DataPreprocessor automatycznie obliczył już te wartości 
        # na podstawie LAT, LON i signaldate. My je tylko pobieramy.
        sog = df["sog_knots"].fillna(0).values
        dcog = df["dcog_deg"].fillna(0).values
        lats = df["LAT"].values
        lons = df["LON"].values
        timestamps = df["signaldate"].values
        
        # Surowa binarna kolumna oznaczająca port
        is_port = df["In Port"].values == 1 if "In Port" in df.columns else np.zeros(len(df), dtype=bool)
        
        # Odległości punkt do punktu dla całej trasy (do wyliczenia wskaźnika prostoliniowości)
        ptp_dists = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
        ptp_dists = np.insert(ptp_dists, 0, 0.0)

        X_sequences = []
        is_port_sequence = []
        indices = []
        
        for i in range(len(df) - window_size + 1):
            w_lats = lats[i : i + window_size]
            w_lons = lons[i : i + window_size]
            
            # Dystans między 1 a ostatnim punktem okna (lepsze wykrywanie kotwicy)
            dist_m = haversine(
                np.array([w_lats[0]]), np.array([w_lons[0]]), 
                np.array([w_lats[-1]]), np.array([w_lons[-1]])
            )[0]
            
            time_delta_seconds = (timestamps[i + window_size - 1] - timestamps[i]) / np.timedelta64(1, 's')
            # Obliczamy średnią prędkość w m/s w całym oknie
            window_speed_ms = dist_m / time_delta_seconds if time_delta_seconds > 0 else 0.0

            # Obliczamy maksymalny rozrzut od środka okna (spread) - specyficzne dla anchor vs adrift
            c_lat, c_lon = w_lats.mean(), w_lons.mean()
            dists_to_center = haversine(w_lats, w_lons, np.full(window_size, c_lat), np.full(window_size, c_lon))
            spread_m = dists_to_center.max()
            
            # Straightness (efficiency) okna: czy statek płynie prosto (Voyage) czy meandruje (Adrift)
            path_dist = np.sum(ptp_dists[i+1 : i+window_size])
            efficiency = dist_m / path_dist if path_dist > 0 else 0.0

            # Suma bezwzględnych zmian kursu w oknie (odróżnia stabilny rejs od chaotycznego dryfowania)
            window_turn_sum = np.sum(np.abs(dcog[i : i + window_size]))

            # Indeks kształtu: odróżnia ruch w linii (voyage/drift) od kręcenia się wokół punktu (anchor)
            shape_index = dist_m / (spread_m + 1.0)

            # Konstruowanie sekwencji: [prędkość, kąt_skrętu, średnia_prędk, spread_m, dystans, prostoliniowość, suma_skrętów, indeks_kształtu]
            seq = []
            for j in range(window_size):
                seq.append([sog[i+j], dcog[i+j], window_speed_ms, spread_m, dist_m, efficiency, window_turn_sum, shape_index])
                
            X_sequences.append(seq)
            
            # Określamy stan dla środkowego punktu okna
            center_idx = i + window_size // 2
            is_port_sequence.append(is_port[center_idx])
            indices.append(center_idx)
            
        return np.array(X_sequences), np.array(is_port_sequence), np.array(indices)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        # 1. Odpalamy preprocesing
        X_seq, is_port_seq, idxs = self._build_features_and_route(train_df)
        
        # Do treningu pobieramy etykiety i dostosowujemy je do naszego zadania
        labels = train_df["operation_id"].values[idxs]
        
        # ==========================================
        # 2. Trening modelu dla operacji MORSKICH
        # ==========================================
        sea_mask = ~is_port_seq
        X_sea = X_seq[sea_mask]
        # W surowym CSV operation_id to: 2=Anchor, 3=Adrift, 4=Voyage. 
        # Odejmując 2, mapujemy to ładnie na indeksy sieci: 0=Anchor, 1=Adrift, 2=Voyage
        y_sea = (labels[sea_mask] - 2).astype(int) 
        
        # --- NOWE: Wyliczanie wag klas (balansowanie z powodu małej ilości adrift) ---
        counts = np.bincount(y_sea, minlength=3)
        counts = np.maximum(counts, 1)  # unikamy dzielenia przez 0
        # Zmniejszamy siłę wag dla rzadkich klas (używamy potęgi 0.25 zamiast 0.5).
        # Sieć wykrywa dryf świetnie (92% recall), ale czasami nadużywa go przy kotwicy.
        # Mniejsza waga uciszy "panikę" sieci i zauważalnie podniesie Precyzję klasy adrift.
        weights = 1.0 / np.power(counts, 0.25)
        weights = (weights / np.sum(weights)) * 3.0  # normalizujemy do ilości klas
        # -----------------------------------------------------------------------------
        
        epochs = self.config.model.params.get("epochs", 25)
        sea_model = SimpleSeaModelWrapper(epochs=epochs, class_weights=weights)
        
        # NOWE: Standaryzacja cech! Sieć LSTM potrzebuje znormalizowanych danych (mean=0, std=1)
        scaler = StandardScaler()
        if len(X_sea) > 0:
            N, W, F = X_sea.shape
            X_sea_scaled = scaler.fit_transform(X_sea.reshape(-1, F)).reshape(N, W, F)
            self.logger.info(f"Trenowanie modelu morskiego na {len(X_sea)} oknach.")
            sea_model.fit(X_sea_scaled, y_sea)
            
        # ==========================================
        # 3. Trening modelu dla manewrów PORTOWYCH
        # ==========================================
        port_mask = is_port_seq
        X_port = X_seq[port_mask]
        
        # Obecnie dla portu operation_id = 1 (Port Stay). 
        y_port = np.zeros(len(X_port), dtype=int)
        
        port_model = PortManeuverNN()
        if len(X_port) > 0:
            self.logger.info(f"Trenowanie modelu portowego na {len(X_port)} oknach.")
            port_model.fit(X_port, y_port)
            
        return {"sea_model": sea_model, "port_model": port_model, "scaler": scaler}

    def predict(self, trained: Any, df: pd.DataFrame) -> pd.DataFrame:
        sea_model = trained["sea_model"]
        port_model = trained["port_model"]
        scaler = trained.get("scaler")
        
        # Preprocesing (Rozdzielacz)
        X_seq, is_port_seq, idxs = self._build_features_and_route(df)
        
        result = df.copy()
        if "predicted_state" not in result.columns:
            result["predicted_state"] = "adrift"
        if "confidence" not in result.columns:
            result["confidence"] = 50.0
        
        # 1. Trasa do sieci Portowej
        if is_port_seq.any():
            result.iloc[idxs[is_port_seq], result.columns.get_loc("predicted_state")] = "port_stay"
            result.iloc[idxs[is_port_seq], result.columns.get_loc("confidence")] = 95.0
            
        # 2. Trasa do sieci Morskiej
        if (~is_port_seq).any():
            # Teraz predict zwraca (predykcje, pewności)
            X_sea = X_seq[~is_port_seq]
            if scaler is not None and len(X_sea) > 0:
                N, W, F = X_sea.shape
                X_sea = scaler.transform(X_sea.reshape(-1, F)).reshape(N, W, F)
            sea_preds, sea_confs = sea_model.predict(X_sea)
            
            SEA_STATE_MAP = {0: "anchor", 1: "adrift", 2: "voyage"}
            pred_strings = [SEA_STATE_MAP.get(p, "adrift") for p in sea_preds]
            
            result.iloc[idxs[~is_port_seq], result.columns.get_loc("predicted_state")] = pred_strings
            result.iloc[idxs[~is_port_seq], result.columns.get_loc("confidence")] = sea_confs
            
        # Uzupełnienie pustych miejsc dla punktów skrajnych okna (bfill/ffill)
        result["predicted_state"] = result["predicted_state"].ffill().bfill()
        result["confidence"] = result["confidence"].ffill().bfill()
        
        return result