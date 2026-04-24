# Deploy this IDS app (fastest → full control)

## Option A — **Fastest: Streamlit Community Cloud** (~2 min)

1. Push this repo to **GitHub** (public is simplest).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) → sign in with GitHub.
3. **New app** → pick repo → main branch → main file: **`app/streamlit_app.py`** → **Deploy**.

**Secrets (Settings → Secrets):**

```toml
IDS_USER = "you@example.com"
IDS_PASS = "your-secure-password"
```

**Notes**

- No Docker build — usually quickest.
- First open: if there is no trained model, use **“Setup demo (download + train)”** in the app (can take a few minutes; if it OOMs, redeploy on a plan with more RAM or train locally and commit `models/` — your `.gitignore` currently ignores `models/`; for Cloud you can remove that ignore for a small demo model, or use only bootstrap).

---

## Option B — **Render (Docker)** (good for a fixed URL + container)

1. [Render](https://dashboard.render.com) → **New** → **Blueprint** → connect GitHub → uses **`render.yaml`** + **`Dockerfile`**.
2. Set **Environment** `IDS_USER` and `IDS_PASS` in the service.

**Notes**

- First Docker build can take **several minutes** (XGBoost + deps).
- Free tier may be tight for training; upgrade RAM if setup demo fails.

---

## Option C — **Local**

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Use `data/test.csv` + `models/ids_model.joblib` locally for instant demo if you already trained.
