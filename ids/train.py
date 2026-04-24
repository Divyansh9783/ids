from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import load_csv
from .pipeline import train_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Train XGBoost IDS model on a CSV dataset.")
    p.add_argument("--data", required=True, help="Path to training CSV")
    p.add_argument("--label-col", default="label", help="Label column name (default: label)")
    p.add_argument("--out", default="models/ids_model.joblib", help="Output model path")
    p.add_argument("--metrics-out", default="models/metrics.json", help="Output metrics JSON")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    p.add_argument("--no-smote", action="store_true", help="Disable SMOTE balancing")
    p.add_argument("--pca", action="store_true", help="Enable PCA dimensionality reduction")
    p.add_argument("--pca-components", type=int, default=20, help="PCA components (default 20)")
    args = p.parse_args()

    df = load_csv(args.data)
    artifacts, metrics = train_eval(
        df,
        label_col=args.label_col,
        test_size=args.test_size,
        use_smote=not args.no_smote,
        use_pca=args.pca,
        pca_components=args.pca_components,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": artifacts.pipeline,
            "feature_columns": artifacts.feature_columns,
            "label_col": artifacts.label_col,
        },
        out_path,
    )

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {out_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("Metrics summary:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "n_train", "n_test"]:
        print(f"- {k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()

