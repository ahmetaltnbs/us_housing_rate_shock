# /src/02_ingest_rates.py
from __future__ import annotations

import pandas as pd

from config import (
    ensure_dirs,
    RAW_DIR,
    RATE_SHOCK_Q_CSV,
    SHOCK_CFG,
)

# FRED public CSV (no key)
FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
RAW_OUT = RAW_DIR / "mortgage30us_weekly_raw.csv"


def main() -> int:
    ensure_dirs()

    # 1) Download weekly series from FRED
    df = pd.read_csv(FRED_GRAPH_CSV)
    # Expected columns: DATE, MORTGAGE30US
    df.columns = [c.strip().lower() for c in df.columns]

    # FRED bazen 'date', bazen 'observation_date' döndürüyor
    date_col = None
    if "date" in df.columns:
        date_col = "date"
    elif "observation_date" in df.columns:
        date_col = "observation_date"

    if date_col is None or "mortgage30us" not in df.columns:
        raise ValueError(f"Beklenmeyen kolonlar: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df[date_col], errors="raise")
    df["rate"] = pd.to_numeric(df["mortgage30us"], errors="coerce")

    df = df.dropna(subset=["rate"]).sort_values("date")

    # Save raw copy (optional ama iyi pratik)
    df[["date", "rate"]].to_csv(RAW_OUT, index=False)
    print(f"[SAVED RAW] {RAW_OUT}")

    # 2) Weekly -> Quarterly (mean)
    # FRED series is weekly; we take quarterly average
    q = (
        df.set_index("date")["rate"]
        .resample("QE")
        .mean()
        .to_frame("rate_q")
        .reset_index()
    )
    q = q.rename(columns={"date": "qdate"})

    # 3) Shock definition
    if SHOCK_CFG.use_rate_change:
        q["shock"] = q["rate_q"].diff()
        shock_label = "dq_rate"
    else:
        w = SHOCK_CFG.ma_window_quarters
        q["rate_ma"] = q["rate_q"].rolling(w, min_periods=w).mean()
        q["shock"] = q["rate_q"] - q["rate_ma"]
        shock_label = f"dev_from_ma{w}q"

    out = q[["qdate", "rate_q", "shock"]].copy()

    # Save processed
    out.to_csv(RATE_SHOCK_Q_CSV, index=False)
    print(f"[RATE] quarterly series saved: {RATE_SHOCK_Q_CSV} | shock={shock_label}")

    # quick info
    print(f"[RATE] date range: {out['qdate'].min().date()} → {out['qdate'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
