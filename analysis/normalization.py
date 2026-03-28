import pandas as pd
import numpy as np


def min_max_normalize(s: pd.Series) -> pd.Series:
    """Normalize series to [0, 100] range using min-max scaling."""
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series(50.0, index=s.index, name=s.name)
    return (s - min_val) / (max_val - min_val) * 100


def z_score_normalize(s: pd.Series) -> pd.Series:
    """Normalize series using z-score (mean=0, std=1)."""
    mean = s.mean()
    std = s.std()
    if std == 0:
        return pd.Series(0.0, index=s.index, name=s.name)
    return (s - mean) / std
