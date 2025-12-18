# /src/01_ingest_hpi.py
from __future__ import annotations

import sys
import pandas as pd
import numpy as np

from config import (
    ensure_dirs,
    HPI_MASTER_CSV,
    HPI_STATE_Q_CSV,
    HPI_MSA_Q_CSV,
    HPI_CFG,
)


REQ_COLS = [
    "hpi_type", "hpi_flavor", "frequency", "level",
    "place_name", "place_id", "yr", "period",
    "index_nsa", "index_sa",
]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"hpi_master.csv eksik sütunlar: {missing}")


def _make_qdate(df: pd.DataFrame) -> pd.DataFrame:
    # period = quarter (1-4) bekliyoruz çünkü quarterly filtreliyoruz
    df["yr"] = pd.to_numeric(df["yr"], errors="raise").astype(int)
    df["period"] = pd.to_numeric(df["period"], errors="raise").astype(int)
    df["qdate"] = pd.PeriodIndex.from_fields(year=df["yr"], quarter=df["period"], freq="Q").to_timestamp()
    return df


def _select_index_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    idx_col = "index_sa" if HPI_CFG.use_sa else "index_nsa"

    # SA seçtiysek ama veri çok boşsa erken uyarı verelim
    null_share = df[idx_col].isna().mean()
    if null_share > 0.05:
        # %5'ten fazla boşluk varsa bu seri muhtemelen SA sağlam gelmemiştir
        raise ValueError(
            f"{idx_col} sütununda boş oranı %{null_share*100:.1f}. "
            f"Bu level/seri için SA yok olabilir. "
            f"Çözüm: 00_config.py içinde HPI_CFG.use_sa=False yapıp tekrar dene."
        )

    return df, idx_col


def _build_panel(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    # numeric + sort
    df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce")

    df = df.sort_values(["place_id", "qdate"]).copy()

    # log level + growth
    df["log_hpi"] = np.log(df[idx_col])
    df["hpi_g"] = 100 * df.groupby("place_id")["log_hpi"].diff()

    # seçili kolonlar
    keep = [
        "place_id", "place_name", "qdate",
        "hpi_type", "hpi_flavor", "frequency", "level",
        idx_col, "log_hpi", "hpi_g"
    ]
    out = df[keep].rename(columns={idx_col: "hpi_index"}).copy()

    return out


def extract_hpi(level_value: str) -> pd.DataFrame:
    """
    level_value: 'State' veya 'MSA'
    """
    df = pd.read_csv(HPI_MASTER_CSV)
    _validate_columns(df)

    # baseline filtre (config)
    df = df[
        (df["hpi_type"] == HPI_CFG.hpi_type) &
        (df["hpi_flavor"] == HPI_CFG.hpi_flavor) &
        (df["frequency"] == HPI_CFG.frequency) &
        (df["level"] == level_value)
    ].copy()

    if df.empty:
        raise ValueError(
            f"Filtre sonrası veri boş geldi. Filtre: "
            f"type={HPI_CFG.hpi_type}, flavor={HPI_CFG.hpi_flavor}, "
            f"freq={HPI_CFG.frequency}, level={level_value}."
        )

    df = _make_qdate(df)
    df, idx_col = _select_index_col(df)
    out = _build_panel(df, idx_col)

    return out


def main() -> int:
    ensure_dirs()

    # 1) State panel
    state = extract_hpi(HPI_CFG.level_state)

    # Beklenen: 50 eyalet + DC = 51 unique place_id
    n_states = state["place_id"].nunique()
    q_min, q_max = state["qdate"].min(), state["qdate"].max()
    print(f"[HPI-STATE] unique place_id: {n_states} | date range: {q_min.date()} → {q_max.date()}")

    state.to_csv(HPI_STATE_Q_CSV, index=False)
    print(f"[SAVED] {HPI_STATE_Q_CSV}")

    # 2) MSA panel (Saiz birleştirmesi için lazım olacak)
    msa = extract_hpi(HPI_CFG.level_msa)
    n_msa = msa["place_id"].nunique()
    q_min2, q_max2 = msa["qdate"].min(), msa["qdate"].max()
    print(f"[HPI-MSA] unique place_id: {n_msa} | date range: {q_min2.date()} → {q_max2.date()}")

    msa.to_csv(HPI_MSA_Q_CSV, index=False)
    print(f"[SAVED] {HPI_MSA_Q_CSV}")

    # Quick sanity check: ilk büyüme değeri her place_id için NaN olacak (diff)
    # Bu normal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
