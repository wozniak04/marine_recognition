from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import ProjectConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)


def load_gps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["signaldate"] = pd.to_datetime(df["signaldate"])
    return df


def load_classified_data(config: ProjectConfig) -> pd.DataFrame:
    """Load and concatenate all labeled GPS CSVs listed in config."""
    frames: list[pd.DataFrame] = []
    for filename in config.data.input_files:
        path = config.paths.raw_dir / filename
        logger.info("Loading %s", path)
        frames.append(load_gps_csv(path))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(df), len(frames))
    return df


def load_ais_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "AIS_signaldate": "signaldate",
        "AIS_LAT": "LAT",
        "AIS_LON": "LON",
        "AIS_SOG": "sog_knots",
        "AIS_COG": "cog_deg",
        "AIS_HDG": "hdg_deg",
        "AIS_DESTINATION_PORT": "destination_port",
    })
    df["signaldate"] = pd.to_datetime(df["signaldate"])
    df = df.sort_values("signaldate").reset_index(drop=True)
    logger.info("Loaded AIS data: %d rows, range %s -> %s", len(df), df["signaldate"].min(), df["signaldate"].max())
    return df


def load_port_database(config: ProjectConfig) -> pd.DataFrame:
    path = config.paths.ports_csv
    logger.info("Loading port database from %s", path)
    df = pd.read_csv(path)
    df = df.dropna(subset=["LAT", "LON"]).reset_index(drop=True)
    return df
