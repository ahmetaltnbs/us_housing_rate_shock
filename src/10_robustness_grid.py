# /src/10_robustness_grid.py
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    ensure_dirs,
    TABLE_DIR,
    PANEL_STATE_Q_CSV,
    SAIZ_STATE_CSV,
    STATE_PERMITS_SENS_CSV,
    RATE_Q_CSV,
)

OUT_XLSX = TABLE_DIR / "robustness_grid.xlsx"
OUT_CSV  = TABLE_DIR / "robustness_grid.csv"


def make_rate_variants(rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input expects at least: qdate + (rate column) + (optional shock column).
    Output: qdate + multiple shock variants.
    """
    r = rate_df.copy()
    r = r.sort_values("qdate")

    # Find/rename rate column robustly
    if "rate" not in r.columns:
        # common patterns
        for c in r.columns:
            lc = c.lower()
            if lc in {"mortgage30us", "mortgage_rate", "mortgage", "rate_level"}:
                r = r.rename(columns={c: "rate"})
                break
        if "rate" not in r.columns:
            # fallback: first numeric-ish column other than qdate/shock
            candidates = [c for c in r.columns if c not in {"qdate", "shock"}]
            for c in candidates:
                if pd.api.types.is_numeric_dtype(r[c]):
                    r = r.rename(columns={c: "rate"})
                    break

    if "rate" not in r.columns:
        raise ValueError(f"RATE_Q_CSV içinde 'rate' kolonu bulunamadı. Kolonlar: {r.columns.tolist()}")

    # Baseline dev_from_ma8q (mevcut shock varsa onu kullan, yoksa üret)
    if "shock" in r.columns:
        r["dev_ma8q"] = r["shock"]
    else:
        r["dev_ma8q"] = r["rate"] - r["rate"].rolling(8).mean()

    r["dev_ma12q"] = r["rate"] - r["rate"].rolling(12).mean()
    r["qoq_change"] = r["rate"].diff(1)
    r["yoy_change"] = r["rate"].diff(4)

    r["qperiod"] = r["qdate"].dt.to_period("Q")
    return r[["qperiod", "dev_ma8q", "dev_ma12q", "qoq_change", "yoy_change"]]



def add_lags(df: pd.DataFrame, col: str, K: int) -> pd.DataFrame:
    out = df.copy()
    for k in range(K + 1):
        out[f"{col}_l{k}"] = out.groupby("place_id")[col].shift(k)
    return out


def fit_state(df_state: pd.DataFrame, y: str, shock_col: str, K: int, hac_lags: int = 4):
    xcols = [f"{shock_col}_l{k}" for k in range(K + 1)]
    d = df_state.dropna(subset=[y] + xcols).copy()
    # çok az gözlem varsa fit etme
    if len(d) < (K + 1) + 20:
        return None
    X = sm.add_constant(d[xcols])
    m = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return m


def cum_effect(model, shock_col: str, horizon: int):
    idx = [f"{shock_col}_l{k}" for k in range(horizon)]
    betas = np.array([model.params.get(i, np.nan) for i in idx], dtype=float)

    # covariance slice
    V = model.cov_params()
    V = V.loc[idx, idx].to_numpy(dtype=float)

    v = np.ones(horizon)
    var = float(v @ V @ v)
    se = np.sqrt(var) if var >= 0 else np.nan
    cum = float(np.nansum(betas))
    t = cum / se if (se is not None and np.isfinite(se) and se > 0) else np.nan
    return cum, se, t


def cross_state_regs(df: pd.DataFrame):
    """
    Returns list of dicts. If too few obs, returns NaN rows (no crash).
    """
    def ols(y, Xcols, name):
        d = df.dropna(subset=[y] + Xcols).copy()
        out = {"model": name}

        if len(d) < 10:
            out.update({"nobs": int(len(d)), "r2": np.nan})
            for v in ["const"] + Xcols:
                out[f"b_{v}"] = np.nan
                out[f"se_{v}"] = np.nan
                out[f"p_{v}"] = np.nan
            return out

        X = sm.add_constant(d[Xcols])
        m = sm.OLS(d[y], X).fit(cov_type="HC1")

        out.update({"nobs": int(m.nobs), "r2": float(m.rsquared)})
        for v in ["const"] + Xcols:
            out[f"b_{v}"] = float(m.params.get(v))
            out[f"se_{v}"] = float(m.bse.get(v))
            out[f"p_{v}"] = float(m.pvalues.get(v))
        return out

    rows = []
    rows.append(ols("price_sens_1y", ["saiz_elasticity"], "price~saiz"))
    rows.append(ols("price_sens_1y", ["permits_sens_1y"], "price~permits"))
    rows.append(ols("price_sens_1y", ["saiz_elasticity", "permits_sens_1y"], "price~saiz+permits"))
    return rows


def main() -> int:
    ensure_dirs()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_STATE_Q_CSV)
    panel["qdate"] = pd.to_datetime(panel["qdate"])
    panel = panel.sort_values(["place_id", "qdate"]).copy()

    # Y kolonunu doğrula (panelde farklı isim varsa burada yakalayalım)
    if "hpi_g" not in panel.columns:
        raise ValueError(
            f"panel dosyasında 'hpi_g' yok. Kolonlar: {panel.columns.tolist()}\n"
            "Muhtemelen HPI büyüme kolonunun adı farklı; 05_estimate_sensitivities.py'deki isimle aynı olmalı."
        )

    perm_s = pd.read_csv(STATE_PERMITS_SENS_CSV).rename(columns={"cum_1y": "permits_sens_1y"})
    saiz_s = pd.read_csv(SAIZ_STATE_CSV)

    rate = pd.read_csv(RATE_Q_CSV)
    rate["qdate"] = pd.to_datetime(rate["qdate"])
    rv = make_rate_variants(rate)

    panel["qperiod"] = panel["qdate"].dt.to_period("Q")

    base = panel.merge(rv, on="qperiod", how="left")

    miss_any = base[["dev_ma8q", "dev_ma12q", "qoq_change", "yoy_change"]].isna().mean()
    print("[MERGE] missing share by shock:\n", miss_any.to_string())

    shock_vars = ["dev_ma8q", "dev_ma12q", "qoq_change", "yoy_change"]
    Ks = [4, 6, 8]
    horizon = 4
    hac_lags = 4

    results = []

    for shock in shock_vars:
        for K in Ks:
            df = base.copy()
            df = add_lags(df, shock, K)

            rows = []
            for pid, g in df.groupby("place_id", sort=True):
                m = fit_state(g, y="hpi_g", shock_col=shock, K=K, hac_lags=hac_lags)
                if m is None:
                    continue
                cum, se, t = cum_effect(m, shock, horizon=horizon)
                rows.append({"place_id": pid, "price_sens_1y": cum, "se_1y": se, "t_1y": t})

            sens = pd.DataFrame(rows)

            # sens boş olsa bile kolonları garanti et (KeyError fix)
            if sens.empty:
                sens = pd.DataFrame(columns=["place_id", "price_sens_1y", "se_1y", "t_1y"])

            cs = (
                sens.merge(perm_s[["place_id", "permits_sens_1y"]], on="place_id", how="left")
                    .merge(saiz_s, on="place_id", how="left")
            )

            regs = cross_state_regs(cs)

            states_in_sens = int(sens["place_id"].nunique()) if "place_id" in sens.columns else 0
            states_with_saiz = int(cs.dropna(subset=["saiz_elasticity"])["place_id"].nunique()) if len(cs) else 0

            for r in regs:
                r.update({
                    "shock_def": shock,
                    "K_lags": K,
                    "horizon_q": horizon,
                    "states_in_sens": states_in_sens,
                    "states_with_saiz": states_with_saiz,
                })
                results.append(r)

            print(f"[OK] shock={shock} | K={K} | states_sens={states_in_sens} | saiz_states={states_with_saiz}")

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"[SAVED] {OUT_CSV}")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="robustness_grid", index=False)
    print(f"[SAVED] {OUT_XLSX}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
