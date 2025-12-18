# /src/00_config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# -------------------------
# Project paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../project_us_house
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RATE_Q_CSV = PROCESSED_DIR / "rate_shock_quarterly.csv"

FIG_DIR = PROCESSED_DIR / "figures"
MAP_DIR = PROCESSED_DIR / "maps"
TABLE_DIR = PROCESSED_DIR / "tables"
CACHE_DIR = PROCESSED_DIR / "_cache"


def ensure_dirs() -> None:
    """Create output folders if missing."""
    for p in [PROCESSED_DIR, FIG_DIR, MAP_DIR, TABLE_DIR, CACHE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Input filenames (raw)
# -------------------------
HPI_MASTER_CSV = RAW_DIR / "hpi_master.csv"
SAIZ_DTA = RAW_DIR / "saiz_elasticities.dta"

# Bu ikisini sen indirince /raw içine koyacağız:
MORTGAGE_RATE_CSV = RAW_DIR / "mortgage_rate.csv"
PERMITS_CSV = RAW_DIR / "permits.csv"


# -------------------------
# Output filenames (processed)
# -------------------------
HPI_STATE_Q_CSV = PROCESSED_DIR / "hpi_state_quarterly.csv"
HPI_MSA_Q_CSV = PROCESSED_DIR / "hpi_msa_quarterly.csv"

RATE_SHOCK_Q_CSV = PROCESSED_DIR / "rate_shock_quarterly.csv"
PERMITS_STATE_Q_CSV = PROCESSED_DIR / "permits_state_quarterly.csv"

PANEL_STATE_Q_CSV = PROCESSED_DIR / "panel_state_quarterly.csv"

STATE_PRICE_SENS_CSV = PROCESSED_DIR / "state_price_sensitivity.csv"
STATE_PERMITS_SENS_CSV = PROCESSED_DIR / "state_permits_sensitivity.csv"

SAIZ_STATE_CSV = PROCESSED_DIR / "saiz_state_aggregated.csv"
REG_RESULTS_CSV = TABLE_DIR / "reg_results.csv"


# -------------------------
# Research settings
# -------------------------
@dataclass(frozen=True)
class HPISettings:
    # En temiz baseline seri:
    hpi_type: str = "traditional"
    hpi_flavor: str = "purchase-only"
    frequency: str = "quarterly"
    use_sa: bool = True  # True => index_sa, False => index_nsa

    # state-level filtre:
    level_state: str = "State"
    # MSA serileri dosyada "MSA" diye geçiyor
    level_msa: str = "MSA"


@dataclass(frozen=True)
class ShockSettings:
    # “Beklenmedik artış” için basit ve şeffaf: rate - MA(rate, window)
    ma_window_quarters: int = 8  # 2 yıl = 8 çeyrek
    # İstersen alternatif şok tanımı (değişim) ekleriz:
    use_rate_change: bool = False  # True => Δrate, False => deviation-from-trend


@dataclass(frozen=True)
class EstimationSettings:
    # distributed lag uzunluğu
    max_lag_quarters: int = 8  # 2 yıl gecikme
    # 1 yıllık etki = ilk 4 katsayının toplamı
    cum_horizon_quarters: int = 4
    # HAC / Newey-West için max lag (genelde 4 veya 8 mantıklı)
    hac_lags: int = 4


HPI_CFG = HPISettings()
SHOCK_CFG = ShockSettings()
EST_CFG = EstimationSettings()
