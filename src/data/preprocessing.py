import pandas as pd
import numpy as np

from src.utils.geo import haversine, calc_course


class DataPreprocessor:
    """
    Cleans and transforms raw GPS data into kinematic features
    including distance, velocity, acceleration, and course over ground.
    """
    def __init__(self, drop_prev: bool = True, drift_threshold: int = 0.51444, clean_data: bool = True):
        self.drop_prev = drop_prev
        self.drift_threshold = drift_threshold

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run pipeline method.
        """
        return (
            df.copy()
            .pipe(self._clean_data)
            .pipe(self._calculate_kinematics)
            .pipe(self._handle_outliers_and_drift)
        )
    
    # ============ DATA CLEANING ============
    
    def _clean_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data by removing invalid coordinates, duplicates and sorting by time. 
        Flags first record.
        """
        
        # If NaN in GPS => remove
        frame = frame.dropna(subset=["LAT", "LON"]).reset_index(drop=True)
        
        # Drop invalid GPS 
        frame = frame[frame["LAT"].between(-90, 90) & frame["LON"].between(-180, 180)]
        frame = frame[(frame["LAT"] != 0) & (frame["LON"] != 0)]
        
        # Clean date
        frame["signaldate"] = pd.to_datetime(frame["signaldate"])
        frame = frame.drop_duplicates(subset=["signaldate"])
        frame = frame.sort_values(["signaldate"]).reset_index(drop=True)
        
        # Is first row
        frame.insert(0, "is_first", frame.index == 0)
            
        return frame

    # ============ KINEMATICS PIPELINE ============

    def _calculate_kinematics(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Computes movement metrics: distance, time delta, velocity, acceleration and COG.
        """
        frame = self._add_prev_values(frame)
        frame = self._calculate_distance_dtime(frame)
        frame = self._calculate_velocity(frame)
        frame = self._calculate_COG(frame)
        frame = self._remove_drift_noise(frame)
        frame = self._calculace_kinematics_params(frame)

        if self.drop_prev:
            frame = self._drop_helper_columns(frame)

        return frame

    def _calculace_kinematics_params(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates movement metrics parameters: dVelocity, acceleration, dCOG.
        """
        # Velocity diff
        frame["prev_velocity_m_s"] = frame["velocity_m_s"].shift(1)
        frame["dVelocity_m_s"] = frame["velocity_m_s"] - frame["prev_velocity_m_s"]
        
        # COG diff
        frame["prev_COG"] = frame["COG"].shift(1)
        frame["dCOG"] = (frame["COG"] - frame["prev_COG"] + 180) % 360 - 180 # normalization (-180* : 180*)
        frame["dCOG"] = frame["dCOG"].fillna(0.0)
        
        # Acceleration
        frame["acc"] = frame["dVelocity_m_s"] / frame["dt"]
    
        return frame
        
    
    # Add previous LAT, LON and signaldate
    def _add_prev_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["prev_LAT"] = frame["LAT"].shift(1)
        frame["prev_LON"] = frame["LON"].shift(1)
        frame["prev_signaldate"] = frame["signaldate"].shift(1)
        
        return frame

    # Calculate traveled distance in meters
    def _calculate_distance_dtime(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["traveled"] = haversine(
            frame["prev_LAT"].values,
            frame["prev_LON"].values,
            frame["LAT"].values,
            frame["LON"].values
        )

        frame["dt"] = (frame["signaldate"] - frame["prev_signaldate"]).dt.total_seconds()

        return frame
    
    # Calculate COG
    def _calculate_COG(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["COG"] = calc_course(
            frame["prev_LAT"].values,
            frame["prev_LON"].values,
            frame["LAT"].values,
            frame["LON"].values
        )
    
        return frame
        

    def _calculate_velocity(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["velocity_m_s"] = frame["traveled"] / frame["dt"]
        frame["velocity_knot"] = frame["velocity_m_s"] * 1.94384
        
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
    
    
    
    # ============ OUTLIERS & DRIFT ============
    def _remove_drift_noise(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Noise filter : velocity < drift_threshold
        frame["is_drift"] = (frame["velocity_m_s"] < self.drift_threshold).astype(int)

        # If drifing, set velocity to 0, course = NaN
        is_drift = frame["is_drift"] == 1
        frame.loc[is_drift, ["velocity_m_s", "velocity_knot", "traveled"]] = 0.0
        frame.loc[is_drift, ["COG"]] = np.nan
        
        return frame 
        
    def _handle_outliers_and_drift(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers from data using IQR and flags GPS drift.
        """
        
        columns_to_check = ["velocity_m_s"]
        
        for col in columns_to_check:
            Q1 = frame[col].quantile(0.25)
            Q3 = frame[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR 
            upper_bound = Q3 + 1.5 * IQR
            
            frame = frame[frame[col].between(lower_bound, upper_bound)]

        # Uśrednia pozycję z 3 ostatnich minut, wygładzając skoki GPS
        # a = frame["LAT"].rolling(window=3, center=True, min_periods=1).mean()
        # b = frame["LON"].rolling(window=3, center=True, min_periods=1).mean()
        
        # frame.insert(3, "LAT_smooth", a)
        # frame.insert(4, "LON_smooth", b)

        return frame