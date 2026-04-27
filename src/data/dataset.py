import pandas as pd
from pathlib import Path

def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)