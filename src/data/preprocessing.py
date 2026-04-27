import pandas as pd

from src.utils.geo import haversine, calc_course

def signaldate_conv(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    transformed["signaldate"] = pd.to_datetime(transformed["signaldate"])
    return transformed

def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    transformed["velocity"] = 0.0
    
    transformed = transformed.sort_values("signaldate").reset_index(drop=True)
    
    transformed["prev_LAT"] = transformed["LAT"].shift(1)
    transformed["prev_LON"] = transformed["LON"].shift(1)
    transformed["prev_signaldate"] = transformed["signaldate"].shift(1)
    
    # Traveled distance in meters
    transformed["traveled"] = haversine(
        transformed["prev_LAT"].values,
        transformed["prev_LON"].values,
        transformed["LAT"].values,
        transformed["LON"].values
    )
    
    # Time diff
    transformed["dt"] = (transformed["signaldate"] - transformed["prev_signaldate"]).dt.total_seconds()
    
    # Velocity
    transformed["velocity_m_s"] = transformed["traveled"] / transformed["dt"]
    transformed["velocity_knot"] = transformed["velocity_m_s"] * 1.94384
    
    # Velocity diff
    transformed["prev_velocity_m_s"] = transformed["velocity_m_s"].shift(1)
    transformed["dvelocity_m_s"] = transformed["velocity_m_s"] - transformed["prev_velocity_m_s"]
    
    # Course
    transformed["COG"] = calc_course(
        transformed["prev_LAT"].values,
        transformed["prev_LON"].values,
        transformed["LAT"].values,
        transformed["LON"].values
    )
    
    transformed = transformed.drop(columns=[
        "prev_LAT",
        "prev_LON",
        "prev_signaldate",
        "prev_velocity_m_s"
    ])
    
    transformed.loc[0, ["traveled", "dt", "velocity_m_s", "velocity_knot"]] = 0.0
    
    return transformed