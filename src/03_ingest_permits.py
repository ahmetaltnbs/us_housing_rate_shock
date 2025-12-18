# /src/03_ingest_permits.py
from __future__ import annotations

import pandas as pd
import numpy as np

from config import (
    ensure_dirs,
    HPI_STATE_Q_CSV,
    PERMITS_STATE_Q_CSV,
)

FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Önce NSA dene (log/growth için daha güvenli), yoksa SA'ya düş
PRIMARY_SUFFIX = "BPPRIV"     # Not Seasonally Adjusted
FALLBACK_SUFFIX = "BPPRIVSA"  # Seasonally Adjusted


def _read_fred_series(series_id: str) -> pd.DataFrame:
    url = FRED_GRAPH.format(series_id=series_id)
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]

    # FRED bazen DATE bazen observation_date döndürüyor
    date_col = "date" if "date" in df.columns else ("observation_date" if "observation_date" in df.columns else None)
    if date_col is None:
        raise ValueError(f"{series_id} kolonları beklenmedik: {df.columns.tolist()}")

    # ikinci kolon genelde series_id'nin kendisi (lowercase)
    val_col = None
    for c in df.columns:
        if c != date_col:
            val_col = c
            break
    if val_col is None:
        raise ValueError(f"{series_id} value kolonu bulunamadı")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="raise"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna(subset=["value"]).sort_values("date")

    return out


def main() -> int:
    ensure_dirs()

    # 1) HPI state panelden state listesi çek (50+DC)
    hpi = pd.read_csv(HPI_STATE_Q_CSV)
    states = sorted(hpi["place_id"].dropna().unique().tolist())
    print(f"[PERMITS] states from HPI: {len(states)} -> {states[:10]}...")

    rows = []
    failed = []

    # 2) Her state için FRED’den permits indir
    for st in states:
        sid1 = f"{st}{PRIMARY_SUFFIX}"
        sid2 = f"{st}{FALLBACK_SUFFIX}"

        df = None
        used = None
        try:
            df = _read_fred_series(sid1)
            used = sid1
        except Exception:
            try:
                df = _read_fred_series(sid2)
                used = sid2
            except Exception as e:
                failed.append((st, str(e)))
                continue

        df["place_id"] = st
        df["series_id"] = used
        rows.append(df)
        print(f"[OK] {st}: {used} | obs={len(df)}")

    if failed:
        print("[WARN] failed states:")
        for st, msg in failed:
            print(f"  - {st}: {msg}")

    if not rows:
        raise RuntimeError("Hiç permits serisi indirilemedi.")

    permits_m = pd.concat(rows, ignore_index=True)

    # 3) Monthly -> Quarterly (SUM)
    permits_m["qper"] = permits_m["date"].dt.to_period("Q")
    q = (
        permits_m.groupby(["place_id", "qper"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "permits_q"})
    )

    # 4) qdate = quarter start (HPI ile uyumlu)
    q["qdate"] = q["qper"].dt.start_time
    q = q.sort_values(["place_id", "qdate"])

    # 5) Growth (log1p ile 0 değerine dayanıklı)
    q["log_permits"] = np.log1p(q["permits_q"])
    q["permits_g"] = 100 * q.groupby("place_id")["log_permits"].diff()

    out = q[["place_id", "qdate", "permits_q", "permits_g"]].copy()
    out.to_csv(PERMITS_STATE_Q_CSV, index=False)

    print(f"[SAVED] {PERMITS_STATE_Q_CSV}")
    print(f"[PERMITS] date range: {out['qdate'].min().date()} → {out['qdate'].max().date()}")
    print(f"[PERMITS] states saved: {out['place_id'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
