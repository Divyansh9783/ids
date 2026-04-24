from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
import subprocess

# Ensure project root is importable when running via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ids.live import scapy_sniff_available, simulate_stream
from ids.pipeline import predict_df


APP_TITLE = "ML-based Intrusion Detection System (IDS)"


def _get_env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def check_login() -> bool:
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    st.title(APP_TITLE)
    st.subheader("Authentication")

    user = st.text_input("Username", value="", autocomplete="username")
    pw = st.text_input("Password", value="", type="password", autocomplete="current-password")

    expected_user = _get_env("IDS_USER", "admin")
    expected_pw = _get_env("IDS_PASS", "admin123")

    if st.button("Login"):
        if user == expected_user and pw == expected_pw:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Invalid username/password.")
    st.info("Default login: admin / admin123 (change via IDS_USER, IDS_PASS env vars).")
    return False


def load_model(model_path: str) -> Any:
    bundle = joblib.load(model_path)
    return bundle["pipeline"]


def sidebar_model_picker() -> str:
    st.sidebar.header("Model")
    default = "models/ids_model.joblib"
    model_path = st.sidebar.text_input("Model path", value=default)
    if not Path(model_path).exists():
        st.sidebar.warning("Model file not found. Train first using `python -m ids.train ...`.")
    return model_path


def maybe_bootstrap_demo_assets(model_path: str) -> None:
    """
    Streamlit Community Cloud doesn't persist local artifacts unless committed.
    If the model is missing, offer a one-click bootstrap that downloads NSL-KDD,
    prepares `data/train.csv` + `data/test.csv`, and trains the model.
    """
    if Path(model_path).exists():
        return

    st.error("Model not found.")
    st.info(
        "To make the app work end-to-end on Streamlit Cloud, we can bootstrap the demo "
        "dataset and train the model automatically."
    )
    if st.button("Setup demo (download dataset + train model)"):
        with st.status("Setting up demo assets...", expanded=True) as status:
            try:
                cmd = [sys.executable, "scripts/setup_demo.py"]
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                st.code(proc.stdout or "Done.")
                status.update(label="Demo setup completed.", state="complete")
                st.success("Model trained. Reloading...")
                st.rerun()
            except subprocess.CalledProcessError as e:
                status.update(label="Demo setup failed.", state="error")
                st.code((e.stdout or "") + "\n" + (e.stderr or ""))
                st.error("Setup failed. Check logs above.")


def render_metrics() -> None:
    st.header("Dashboard")
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        c2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        c3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        c4.metric("F1", f"{metrics.get('f1', 0):.4f}")

        if metrics.get("roc_auc") is not None:
            st.caption(f"ROC-AUC: {metrics.get('roc_auc'):.6f}")

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, list) and len(cm) == 2:
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Attack"], columns=["Pred Normal", "Pred Attack"])
            fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        rep = metrics.get("classification_report")
        if rep:
            with st.expander("Classification report"):
                st.code(rep)
    else:
        st.info("No metrics found at `models/metrics.json` yet.")


def _load_default_demo_df() -> Optional[pd.DataFrame]:
    demo_path = Path("data/test.csv")
    if demo_path.exists():
        return pd.read_csv(demo_path)
    return None


def render_upload_and_scan(pipe: Any) -> None:
    st.header("Upload Dataset to Scan Attacks")
    demo_df = _load_default_demo_df()
    if demo_df is not None:
        st.success("Demo dataset found at `data/test.csv`. You can use it without uploading.")
        use_demo = st.toggle("Use demo dataset (`data/test.csv`)", value=True)
    else:
        use_demo = False

    up = None if use_demo else st.file_uploader("Upload CSV", type=["csv"])
    label_col = st.text_input("Label column (optional, will be ignored)", value="")

    if not use_demo and up is None:
        st.caption("Upload a CSV similar to NSL-KDD / CICIDS2017 feature tables.")
        return

    df = demo_df if use_demo else pd.read_csv(up)  # type: ignore[arg-type]
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    if st.button("Run Detection"):
        out = predict_df(pipe, df, label_col=(label_col.strip() or None))

        st.subheader("Results (first rows)")
        st.dataframe(out.head(200), use_container_width=True)

        counts = out["prediction"].value_counts().reset_index()
        counts.columns = ["prediction", "count"]
        fig = px.bar(counts, x="prediction", y="count", title="Predicted Traffic Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary")
        total = int(len(out))
        attacks = int((out["prediction"] == "attack").sum())
        normals = total - attacks
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", f"{total}")
        c2.metric("Normal", f"{normals}")
        c3.metric("Attack", f"{attacks}")

        if "label" in out.columns:
            try:
                y_true = out["label"].astype(str).str.lower().map({"normal": 0, "attack": 1})
                y_pred = out["prediction"].astype(str).str.lower().map({"normal": 0, "attack": 1})
                if y_true.notna().all() and y_pred.notna().all():
                    cm_local = confusion_matrix(y_true, y_pred)
                    cm_df = pd.DataFrame(
                        cm_local,
                        index=["Actual Normal", "Actual Attack"],
                        columns=["Pred Normal", "Pred Attack"],
                    )
                    fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix (if labels present)")
                    st.plotly_chart(fig_cm, use_container_width=True)
            except Exception:
                pass

        if "attack_probability" in out.columns:
            fig2 = px.histogram(
                out,
                x="attack_probability",
                nbins=40,
                title="Attack probability distribution",
            )
            st.plotly_chart(fig2, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="ids_results.csv",
            mime="text/csv",
        )


def render_live_monitoring(pipe: Any) -> None:
    st.header("Live Attack Monitoring")
    st.caption("Default mode is simulation (safe). Packet capture requires root + scapy.")

    mode = st.radio(
        "Live mode",
        options=["Simulation"],
        index=0,
        horizontal=True,
    )

    rate = st.slider("Events/sec", min_value=1, max_value=10, value=2)
    st.info(f"Scapy available: {scapy_sniff_available()}")

    demo_df = _load_default_demo_df()
    if demo_df is not None:
        use_demo = st.toggle("Use demo dataset schema (`data/test.csv`)", value=True, key="live_demo")
    else:
        use_demo = False

    schema_up = None if use_demo else st.file_uploader(
        "Upload a CSV to infer schema for simulation", type=["csv"], key="schema"
    )
    if not use_demo and schema_up is None:
        st.warning("Upload any CSV (same columns as your training data) to start simulation.")
        return

    schema_df = (demo_df if use_demo else pd.read_csv(schema_up)).head(500)  # type: ignore[arg-type]
    placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.button("Start Live Stream"):
        events = []
        gen = simulate_stream(pipe, schema_df=schema_df, rate_per_sec=float(rate))
        for _ in range(200):  # bounded loop for Streamlit run
            ev = next(gen)
            events.append(
                {
                    "ts": ev.ts,
                    "prediction": ev.prediction,
                    "attack_probability": ev.attack_probability,
                }
            )
            df_ev = pd.DataFrame(events)
            placeholder.dataframe(df_ev.tail(20), use_container_width=True)
            fig = px.line(df_ev, x="ts", y="attack_probability", color="prediction", title="Live attack probability")
            chart_placeholder.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_login():
        return

    model_path = sidebar_model_picker()
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Home", "Upload & Scan", "Live Monitoring"],
        index=0,
    )

    if not Path(model_path).exists():
        st.title(APP_TITLE)
        maybe_bootstrap_demo_assets(model_path)
        st.code("Manual train: python -m ids.train --data data/train.csv --label-col label --out models/ids_model.joblib")
        return

    pipe = load_model(model_path)

    st.title(APP_TITLE)

    if page == "Home":
        render_metrics()
    elif page == "Upload & Scan":
        render_upload_and_scan(pipe)
    elif page == "Live Monitoring":
        render_live_monitoring(pipe)


if __name__ == "__main__":
    main()

