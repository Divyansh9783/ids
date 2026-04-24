from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import pandas as pd

NSL_TRAIN_URL = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Train.csv"
NSL_TEST_URL = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Test.csv"

COLS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_type",
]


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    out_path.write_bytes(data)


def convert_raw_nsl_kdd(raw_path: Path, out_path: Path) -> None:
    df = pd.read_csv(raw_path, header=None)
    if df.shape[1] != len(COLS):
        raise SystemExit(f"Unexpected columns in {raw_path}: {df.shape[1]} (expected {len(COLS)})")
    df.columns = COLS
    df["label"] = df["attack_type"].apply(
        lambda x: "normal" if str(x).strip().lower() == "normal" else "attack"
    )
    df = df.drop(columns=["attack_type"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def run_train(project_root: Path) -> None:
    # Run training via module to reuse existing pipeline.
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "ids.train",
        "--data",
        str(project_root / "data" / "train.csv"),
        "--label-col",
        "label",
        "--out",
        str(project_root / "models" / "ids_model.joblib"),
        "--metrics-out",
        str(project_root / "models" / "metrics.json"),
    ]
    subprocess.check_call(cmd, cwd=str(project_root))


def main() -> None:
    p = argparse.ArgumentParser(description="Download NSL-KDD, prepare CSVs, and train the model.")
    p.add_argument("--skip-download", action="store_true", help="Skip downloading raw files")
    p.add_argument("--skip-train", action="store_true", help="Skip training step")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    raw_train = data_dir / "nsl_kdd_train.csv"
    raw_test = data_dir / "nsl_kdd_test.csv"

    if not args.skip_download:
        print("Downloading NSL-KDD raw CSVs...")
        download(NSL_TRAIN_URL, raw_train)
        download(NSL_TEST_URL, raw_test)

    print("Converting to train/test with label column...")
    convert_raw_nsl_kdd(raw_train, data_dir / "train.csv")
    convert_raw_nsl_kdd(raw_test, data_dir / "test.csv")

    if not args.skip_train:
        models_dir.mkdir(parents=True, exist_ok=True)
        print("Training model...")
        run_train(project_root)

    print("Done.")
    print(f"- Train CSV: {data_dir / 'train.csv'}")
    print(f"- Test CSV:  {data_dir / 'test.csv'}")
    print(f"- Model:     {models_dir / 'ids_model.joblib'}")
    print(f"- Metrics:   {models_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

