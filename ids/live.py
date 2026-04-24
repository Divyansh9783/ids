from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiveEvent:
    ts: float
    prediction: str
    attack_probability: Optional[float]
    row: Dict[str, Any]


def simulate_stream(
    pipeline: Any,
    *,
    schema_df: pd.DataFrame,
    rate_per_sec: float = 2.0,
    seed: int = 42,
) -> Generator[LiveEvent, None, None]:
    """
    Safe fallback for "live monitoring" when packet capture isn't available.
    Generates synthetic rows with similar column types as schema_df and streams predictions.
    """
    rng = np.random.default_rng(seed)

    numeric_cols = schema_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in schema_df.columns if c not in numeric_cols]

    cat_values: Dict[str, list] = {}
    for c in cat_cols:
        vals = schema_df[c].dropna().astype(str).unique().tolist()
        cat_values[c] = vals if vals else ["unknown"]

    sleep_s = max(0.0, 1.0 / max(rate_per_sec, 0.01))

    while True:
        row: Dict[str, Any] = {}
        for c in numeric_cols:
            col = schema_df[c]
            mu = float(col.dropna().mean()) if col.dropna().shape[0] else 0.0
            sigma = float(col.dropna().std()) if col.dropna().shape[0] else 1.0
            if sigma == 0.0:
                sigma = 1.0
            row[c] = float(rng.normal(mu, sigma))

        for c in cat_cols:
            row[c] = str(rng.choice(cat_values[c]))

        df = pd.DataFrame([row])
        pred = pipeline.predict(df)[0]
        proba = None
        try:
            proba = float(pipeline.predict_proba(df)[:, 1][0])
        except Exception:
            proba = None

        yield LiveEvent(
            ts=time.time(),
            prediction="attack" if int(pred) == 1 else "normal",
            attack_probability=proba,
            row=row,
        )
        time.sleep(sleep_s)


def scapy_sniff_available() -> bool:
    try:
        import scapy.all as scapy  # noqa: F401

        return True
    except Exception:
        return False

