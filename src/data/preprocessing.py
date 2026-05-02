import pandas as pd
import numpy as np

from src.utils.geo import haversine, calc_course


class DataPreprocessor:
    """
    Cleans and transforms raw GPS data into kinematic features
    including distance, velocity, acceleration, and course over ground.
    """
    def __init__(
        self, 
        drop_prev: bool = True, 
        drift_threshold: int = 0.51444,
        max_speed_m_s: int = 12.86, 
        max_acc_m_s2: int = 0.05
        ):              
        self.drop_prev = drop_prev
        self.drift_threshold = drift_threshold
        self.max_speed_m_s = max_speed_m_s
        self.max_acc_m_s2 = max_acc_m_s2

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the preprocessing pipeline.
        """
        return (
            df.copy()
            .pipe(self._gps_find_outliers)
            .pipe(self._handle_kinematics)
            .pipe(self._segment_data)
            .pipe(self._fill_first_lines_nans)
        )
    
    # ============ DATA CLEANING ============
    
    def _fill_first_lines_nans(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame.iloc[0:2] = frame.iloc[0:2].fillna(0.0)
        
        return frame
    
    def _gps_find_outliers(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data by flagging invalid coordinates, duplicates and sorting by time as outliers. 
        """
        
        print("========== WYSZUKIWANIE OUTLIEROW Z GPS ==========")
        
        frame["is_outlier"] = 0
        
        # Sort
        frame = frame.sort_values(["signaldate"])
        
        # If NaN in GPS => remove
        nan_mask = frame["LAT"].isna() | frame["LON"].isna()
        nan_sum = nan_mask.sum()
        frame.loc[nan_mask, "is_outlier"] = 1
        print(f"[Info]: Nowych outlier'ów, przez braki danych GPS (NaN): {nan_sum}")
        
        # Drop invalid GPS 
        range_mask = frame["LAT"].between(-90, 90) & frame["LON"].between(-180, 180)
        wrong_range_sum = (~range_mask & (frame["is_outlier"] == 0)).sum()
        frame.loc[~range_mask, "is_outlier"] = 1
        print(f"[Info]: Nowych outlier'ów, przez GPS poza zakresem (-90, 90) ; (-180, 180): {wrong_range_sum}")
        
        # Point (0, 0)
        zero_point_mask = (frame["LAT"] == 0) & (frame["LON"] == 0)
        zero_point_sum = (zero_point_mask & (frame["is_outlier"] == 0)).sum()
        frame.loc[zero_point_mask, "is_outlier"] = 1
        print(f"[Info]: Nowych outlier'ów, przez rekordy w punkcie (0, 0): {zero_point_sum}")
        
        # Duplicates
        frame["signaldate"] = pd.to_datetime(frame["signaldate"])
        dupe_mask = frame.duplicated(subset=["signaldate"], keep="first")
        dupe_sum = (dupe_mask & (frame["is_outlier"] == 0)).sum()
        frame.loc[dupe_mask, "is_outlier"] = 1
        print(f"[Info]: Nowych outlier'ów, przez duplikaty dat: {dupe_sum}")
        
        # GPS freeze
        freeze_mask = (frame["LAT"] == frame["LAT"].shift(1)) & (frame["LON"] == frame["LON"].shift(1))
        freeze_sum = (freeze_mask & (frame["is_outlier"] == 0)).sum()
        frame.loc[freeze_mask, "is_outlier"] = 1
        print(f"[Info]: Nowych outlier'ów, przez GPS freeze: {freeze_sum}")
        
        # Summary
        print(f"[Info]: Wykryto łącznie: {(frame["is_outlier"] == 1).sum()} outlierów.")
        
        return frame

    # ============ KINEMATICS PIPELINE ============

    def _handle_kinematics(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Computes movement metrics: distance, time delta, velocity, acceleration and COG.
        """
        print(f"========== KINEMATYKA ==========")
        
        frame = self._add_prev_values(frame)
        frame = self._calculate_distance_dtime(frame)
        frame = self._calculate_velocity(frame)
        frame = self._calculate_COG(frame)
        frame = self._mark_unnatural_velocity(frame)
        frame = self._mark_drift(frame)
        frame = self._calculace_kinematics_params(frame)
        frame = self._mark_unnatural_acceleration(frame)

        if self.drop_prev:
            frame = self._drop_helper_columns(frame)

        return frame

    def _calculace_kinematics_params(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates movement metrics parameters: dVelocity, acceleration, dCOG.
        """
        mask = frame["is_outlier"] == 0
        
        # # Velocity diff
        frame.loc[mask, "prev_velocity_m_s"] = frame.loc[mask, "velocity_m_s"].shift(1)
        frame.loc[mask, "dVelocity_m_s"] = frame.loc[mask, "velocity_m_s"] - frame.loc[mask, "prev_velocity_m_s"]
        
        # # COG diff
        frame.loc[mask, "prev_COG"] = frame.loc[mask, "COG"].shift(1)
        frame.loc[mask, "dCOG"] = (frame.loc[mask, "COG"] - frame.loc[mask, "prev_COG"] + 180) % 360 - 180 # normalization (-180* : 180*)
        frame.loc[mask, "dCOG"] = frame.loc[mask, "dCOG"].fillna(0.0)
        
        # # Acceleration
        frame.loc[mask, "acc"] = frame.loc[mask, "dVelocity_m_s"] / frame.loc[mask, "dt"]
        
        # # Angular Velocity
        frame.loc[mask, "omega"] = frame.loc[mask, "dCOG"] / frame.loc[mask, "dt"]

        return frame
        
    
    # Add previous LAT, LON and signaldate
    def _add_prev_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["is_outlier"] == 0
            
        frame.loc[mask, "prev_LAT"] = frame.loc[mask, "LAT"].shift(1)
        frame.loc[mask, "prev_LON"] = frame.loc[mask, "LON"].shift(1)
        frame.loc[mask, "prev_signaldate"] = frame.loc[mask, "signaldate"].shift(1)
        
        
        return frame

    # Calculate traveled distance in meters
    def _calculate_distance_dtime(self, frame: pd.DataFrame) -> pd.DataFrame:        
        mask = frame["is_outlier"] == 0

        frame["COG"] = np.nan 

        frame.loc[mask, "traveled"] = haversine(
            frame.loc[mask, "prev_LAT"].values,
            frame.loc[mask, "prev_LON"].values,
            frame.loc[mask, "LAT"].values,
            frame.loc[mask, "LON"].values
        )
        
        frame.loc[mask, "dt"] = (frame.loc[mask, "signaldate"] - frame.loc[mask, "prev_signaldate"]).dt.total_seconds()

        return frame
    
    # Calculate COG
    def _calculate_COG(self, frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["is_outlier"] == 0
        frame.loc[mask, "COG"] = calc_course(
            frame.loc[mask, "prev_LAT"].values,
            frame.loc[mask, "prev_LON"].values,
            frame.loc[mask, "LAT"].values,
            frame.loc[mask, "LON"].values
        )
    
        return frame
        

    def _calculate_velocity(self, frame: pd.DataFrame) -> pd.DataFrame:        
        mask = frame["is_outlier"] == 0
        
        frame.loc[mask, "velocity_m_s"] = frame.loc[mask, "traveled"] / frame.loc[mask, "dt"]
        frame.loc[mask, "velocity_knot"] = frame.loc[mask, "velocity_m_s"] * 1.94384
        
        return frame
    
    # ============ CLEANUP (if drop_prev=True) ============
    def _drop_helper_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            "prev_LAT",
            "prev_LON",
            "prev_signaldate",
            "prev_velocity_m_s",
            "prev_COG"
        ]        
        return frame.drop(columns=cols_to_drop)
    
    
    
    # ============ DRIFT ============   
    def _mark_drift(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Marks if ship is drifting.
        """
        frame["is_drift"] = 0
        mask = frame["is_outlier"] == 0
        
        drift_mask = frame["velocity_m_s"] < self.drift_threshold
        drift_sum = (mask & drift_mask).sum()
        frame.loc[mask & drift_mask, "is_drift"] = 1
        
        print(f"[Info]: Wykryto {drift_sum} statków dryfujących.")
        
        return frame 
         
    def _mark_unnatural_velocity(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Marks outlier if velocity is unnatural.
        """
        mask = frame["is_outlier"] == 0
        
        unnatural_mask = frame["velocity_m_s"] > self.max_speed_m_s
        unnatural_sum = (unnatural_mask & mask).sum()
        frame.loc[unnatural_mask & mask, "is_outlier"] = 1
        
        print(f"[Info]: Statki z nienaturalną prędkością: {unnatural_sum}")
        
        return frame
        
    def _mark_unnatural_acceleration(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Marks outlier if acceleration is unnatural.
        """
        mask = frame["is_outlier"] == 0
        unnatural_mask = (frame["acc"] > self.max_acc_m_s2) | (frame["acc"] < -self.max_acc_m_s2)
        unnatural_sum = (unnatural_mask & mask).sum()
        frame.loc[unnatural_mask & mask, "is_outlier"] = 1
        
        print(f"[Info]: Statki z nienaturalnym przyspieszeniem: {unnatural_sum}")
        
        return frame
        
    # ============ SEGMENT DATA ============
    def _segment_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["is_outlier"] == 0
        
        first_valid_idx = frame.index[mask][0]
        is_new_segment = mask & ((frame["dt"] > 300) | (frame.index == first_valid_idx))

        cols_to_zero = ["velocity_m_s", "velocity_knot", "dVelocity_m_s", "dCOG", "acc"]
        frame.loc[is_new_segment, cols_to_zero] = 0.0
        frame.loc[is_new_segment, "is_drift"] = 0

        frame.loc[is_new_segment, "COG"] = np.nan
        frame.loc[mask, "COG"] = frame.loc[mask, "COG"].ffill().fillna(0.0)

        frame["is_first_in_segment"] = 0
        frame.loc[is_new_segment, "is_first_in_segment"] = 1

        frame["segment_id"] = is_new_segment.cumsum()
        frame.loc[~mask, "segment_id"] = -1
        
        return frame