# /src/04_merge_panel_state.py
from __future__ import annotations

import pandas as pd

from config import (
    ensure_dirs,
    HPI_STATE_Q_CSV,
    RATE_SHOCK_Q_CSV,
    PERMITS_STATE_Q_CSV,
    PANEL_STATE_Q_CSV,
)


def _to_qstart(s: pd.Series) -> pd.Series:
    """Convert any datetime series to quarter-start timestamps."""
    d = pd.to_datetime(s, errors="raise")
    return d.dt.to_period("Q").dt.start_time


def main() -> int:
    ensure_dirs()

    # 1) Load
    hpi = pd.read_csv(HPI_STATE_Q_CSV)
    rates = pd.read_csv(RATE_SHOCK_Q_CSV)
    permits = pd.read_csv(PERMITS_STATE_Q_CSV)

    # 2) Parse dates & align to quarter-start
    hpi["qdate"] = pd.to_datetime(hpi["qdate"], errors="raise")  # already quarter-start
    rates["qdate"] = _to_qstart(rates["qdate"])                  # make quarter-start
    permits["qdate"] = pd.to_datetime(permits["qdate"], errors="raise")  # already quarter-start

    # 3) Keep only needed cols
    hpi = hpi[[
        "place_id", "place_name", "qdate",
        "hpi_index", "log_hpi", "hpi_g"
    ]].copy()

    rates = rates[["qdate", "rate_q", "shock"]].copy()

    permits = permits[[
        "place_id", "qdate",
        "permits_q", "permits_g"
    ]].copy()

    # 4) Merge: (state,qdate) panel
    panel = (
        hpi.merge(rates, on="qdate", how="left")
           .merge(permits, on=["place_id", "qdate"], how="left")
           .sort_values(["place_id", "qdate"])
           .reset_index(drop=True)
    )

    # 5) Basic sanity checks
    n_states = panel["place_id"].nunique()
    print(f"[PANEL] states: {n_states}")

    # check duplicates
    dup = panel.duplicated(subset=["place_id", "qdate"]).sum()
    if dup:
        raise ValueError(f"[PANEL] (place_id,qdate) duplicate rows: {dup}")

    # show missingness (rates should be mostly complete in modern period)
    miss_rate = panel["shock"].isna().mean()
    miss_perm = panel["permits_q"].isna().mean()
    print(f"[PANEL] missing shock share: {miss_rate:.3f}")
    print(f"[PANEL] missing permits share: {miss_perm:.3f}")

    # 6) Save
    panel.to_csv(PANEL_STATE_Q_CSV, index=False)
    print(f"[SAVED] {PANEL_STATE_Q_CSV}")
    print(f"[PANEL] date range: {panel['qdate'].min().date()} → {panel['qdate'].max().date()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
