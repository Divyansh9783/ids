from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    label_col: str


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def split_features_label(
    df: pd.DataFrame, *, label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. Available columns: {list(df.columns)[:20]}..."
        )
    y = df[label_col]
    X = df.drop(columns=[label_col])
    return X, y


def normalize_label_series(y: pd.Series) -> pd.Series:
    # Accept: {normal, attack}, {benign, malicious}, {0,1}, {True,False}
    if y.dtype == "bool":
        return y.astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "normal": 0,
        "benign": 0,
        "attack": 1,
        "malicious": 1,
        "intrusion": 1,
    }
    y_mapped = y_str.map(mapping)
    if y_mapped.isna().any():
        bad = sorted(y_str[y_mapped.isna()].unique().tolist())[:10]
        raise ValueError(
            f"Unrecognized label values (showing up to 10): {bad}. "
            f"Supported: normal/benign vs attack/malicious or 0/1."
        )
    return y_mapped.astype(int)


def maybe_drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns that are entirely null/empty
    null_all = df.columns[df.isna().all()].tolist()
    if null_all:
        return df.drop(columns=null_all)
    return df


def coerce_infinite_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([float("inf"), float("-inf")], pd.NA)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = coerce_infinite_to_nan(df)
    df = maybe_drop_empty_columns(df)
    return df

