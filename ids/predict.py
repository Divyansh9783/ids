from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import load_csv
from .pipeline import predict_df


def main() -> None:
    p = argparse.ArgumentParser(description="Run IDS predictions on a CSV dataset.")
    p.add_argument("--model", required=True, help="Path to saved joblib model")
    p.add_argument("--data", required=True, help="Path to CSV to scan")
    p.add_argument("--out", default="predictions.csv", help="Output CSV path")
    p.add_argument("--label-col", default=None, help="Optional label column to ignore")
    args = p.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]

    df = load_csv(args.data)
    out_df = predict_df(pipe, df, label_col=args.label_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()

