from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "final_preprocess_M2.csv"


def load_numeric_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATASET)
    numeric = data.select_dtypes(include=np.number).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    return data, numeric


def selected_features(data: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "DC_POWER": ["DC_POWER"],
        "AC_POWER": ["AC_POWER"],
        "MODULE_TEMPERATURE": ["MODULE_TEMPERATURE"],
        "IRRADIATION": ["IRRADIATION"],
    }
    columns = []
    for name, options in aliases.items():
        found = next((option for option in options if option in data.columns), None)
        if found:
            columns.append(found)
    if len(columns) < 3:
        raise ValueError(f"Expected clustering features were not found: {columns}")
    frame = data[columns].apply(pd.to_numeric, errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).fillna(frame.median()).fillna(0)


def scale(frame: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(frame)
