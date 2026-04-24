#!/usr/bin/env python3
"""
Generate synthetic NSL-KDD-shaped CSVs for load testing (KB, MB, multi-GB).
Same columns as data/train.csv (if present) or a built-in schema.

  python scripts/generate_test_datasets.py
  python scripts/generate_test_datasets.py --skip-2gb
  python scripts/generate_test_datasets.py --target-gb 1.0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

# Columns must match training CSV (no attack_type; label is last)
COLUMNS: List[str] = [
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
    "label",
]

PROTO = ["tcp", "udp", "icmp"]
SERVICE = [
    "http", "https", "ftp_data", "smtp", "private", "other", "domain", "ecr_i", "auth",
]
FLAGS = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]


def _random_rows(rng: np.random.Generator, n: int) -> List[List[object]]:
    rows: List[List[object]] = []
    for _ in range(n):
        proto = str(rng.choice(PROTO))
        service = str(rng.choice(SERVICE))
        flag = str(rng.choice(FLAGS))
        src_b = int(rng.integers(0, 100_000))
        dst_b = int(rng.integers(0, 1_000_000))
        land = int(rng.integers(0, 2))
        wrong_fragment = int(rng.integers(0, 4))
        urgent = int(rng.integers(0, 2))
        hot = int(rng.integers(0, 20))
        num_failed = int(rng.integers(0, 5))
        logged_in = int(rng.integers(0, 2))
        num_comp = int(rng.integers(0, 10))
        root_sh = int(rng.integers(0, 2))
        su_a = int(rng.integers(0, 2))
        num_root = int(rng.integers(0, 5))
        nfc = int(rng.integers(0, 10))
        nshells = int(rng.integers(0, 5))
        naf = int(rng.integers(0, 20))
        noc = 0
        is_host = int(rng.integers(0, 2))
        is_guest = int(rng.integers(0, 2))
        count = int(rng.integers(0, 500))
        srv_count = int(rng.integers(0, 500))
        rate_block_a = [float(rng.random()) for _ in range(7)]
        dst_hc = int(rng.integers(0, 300))
        dst_hsvc = int(rng.integers(0, 300))
        rate_block_b = [float(rng.random()) for _ in range(8)]
        label = "normal" if rng.random() < 0.5 else "attack"
        row = [
            int(rng.integers(0, 30_000)),
            proto,
            service,
            flag,
            src_b,
            dst_b,
            land,
            wrong_fragment,
            urgent,
            hot,
            num_failed,
            logged_in,
            num_comp,
            root_sh,
            su_a,
            num_root,
            nfc,
            nshells,
            naf,
            noc,
            is_host,
            is_guest,
            count,
            srv_count,
        ]
        row.extend(rate_block_a)
        row.extend([dst_hc, dst_hsvc])
        row.extend(rate_block_b)
        row.append(label)
        rows.append(row)
    return rows


def write_csv(path: Path, n_rows: int, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        chunk = 5_000
        left = n_rows
        while left > 0:
            take = min(chunk, left)
            for row in _random_rows(rng, take):
                w.writerow(row)
            left -= take


def write_until_size(
    path: Path, target_bytes: int, rng: np.random.Generator, chunk: int = 5_000
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        f.flush()
        while f.tell() < target_bytes:
            for row in _random_rows(rng, chunk):
                w.writerow(row)
            f.flush()
            total = f.tell()
            print(f"  ... {path.name} ~{total / (1024**3):.2f} GB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/synth", help="Output directory")
    p.add_argument(
        "--kb-rows",
        type=int,
        default=50,
        help="Rows for tiny file (tens of KB; increase for larger small test file)",
    )
    p.add_argument("--mb-target", type=float, default=30.0, help="Target size for MB file")
    p.add_argument("--target-gb", type=float, default=2.0, help="Target size for large file (GB)")
    p.add_argument("--skip-2gb", action="store_true", help="Do not generate multi-GB file")
    args = p.parse_args()

    out = Path(args.out_dir)
    rng = np.random.default_rng(42)

    kb_path = out / "synth_small_kb.csv"
    mb_path = out / "synth_medium_mb.csv"
    gb_path = out / "synth_large_2gb.csv"

    print("Small (KB):", kb_path, f"rows={args.kb_rows}")
    write_csv(kb_path, args.kb_rows, rng)
    print("  size:", kb_path.stat().st_size, "bytes")

    mb_bytes = int(args.mb_target * 1024 * 1024)
    print("Medium (MB):", mb_path, f"~{args.mb_target} MB (writes until size >= target)")
    mb_path.parent.mkdir(parents=True, exist_ok=True)
    with mb_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        f.flush()
        while f.tell() < mb_bytes:
            for row in _random_rows(rng, 2_000):
                w.writerow(row)
            f.flush()
    print("  size:", mb_path.stat().st_size, "bytes")

    if args.skip_2gb:
        print("Skipped large file (--skip-2gb).")
        return

    target = int(args.target_gb * 1024**3)
    print("Large:", gb_path, f"~{args.target_gb} GB (this may take a long time, needs free disk).")
    write_until_size(gb_path, target, rng)
    print("  size:", gb_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
