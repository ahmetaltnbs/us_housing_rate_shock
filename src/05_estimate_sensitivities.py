# /src/05_estimate_sensitivities.py
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    ensure_dirs,
    PANEL_STATE_Q_CSV,
    STATE_PRICE_SENS_CSV,
    STATE_PERMITS_SENS_CSV,
    EST_CFG,
)


def make_lags(df: pd.DataFrame, col: str, max_lag: int) -> pd.DataFrame:
    """Create lag columns col_l0 ... col_lK within each state."""
    out = df.copy()
    for k in range(0, max_lag + 1):
        out[f"{col}_l{k}"] = out.groupby("place_id")[col].shift(k)
    return out


def fit_state_dlag(df_state: pd.DataFrame, y_col: str, x_base: str, max_lag: int, hac_lags: int):
    """
    y_t = a + sum_{k=0..K} b_k * x_{t-k} + e_t
    Returns params and HAC standard errors.
    """
    x_cols = [f"{x_base}_l{k}" for k in range(0, max_lag + 1)]
    d = df_state.dropna(subset=[y_col] + x_cols).copy()
    if len(d) < (max_lag + 1) + 15:
        return None  # too few obs

    X = sm.add_constant(d[x_cols])
    y = d[y_col]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model


def summarize_cum_effect(model, x_base: str, horizon: int):
    """
    cumulative effect = sum_{k=0..horizon-1} b_k
    Also approximate SE using covariance matrix.
    """
    betas = np.array([model.params.get(f"{x_base}_l{k}", np.nan) for k in range(horizon)])
    # Var(sum b) = 1' V 1
    V = model.cov_params()
    idx = [f"{x_base}_l{k}" for k in range(horizon)]
    # Build vector
    v = np.ones(horizon)
    # Extract covariance submatrix for selected betas
    Vsub = V.loc[idx, idx].to_numpy()
    var_cum = float(v @ Vsub @ v)
    se_cum = np.sqrt(var_cum) if var_cum >= 0 else np.nan

    cum = float(np.nansum(betas))
    t = cum / se_cum if se_cum and se_cum > 0 else np.nan
    return cum, se_cum, t


def main() -> int:
    ensure_dirs()

    df = pd.read_csv(PANEL_STATE_Q_CSV)
    df["qdate"] = pd.to_datetime(df["qdate"], errors="raise")
    df = df.sort_values(["place_id", "qdate"]).copy()

    # 1) Build shock lags per state
    df = make_lags(df, col="shock", max_lag=EST_CFG.max_lag_quarters)

    # 2) Run state-by-state models
    price_rows = []
    perm_rows = []

    for pid, g in df.groupby("place_id", sort=True):
        g = g.copy()

        # PRICE: use hpi_g as dependent variable
        m_price = fit_state_dlag(
            g, y_col="hpi_g", x_base="shock",
            max_lag=EST_CFG.max_lag_quarters,
            hac_lags=EST_CFG.hac_lags
        )

        if m_price is not None:
            cum, se, t = summarize_cum_effect(m_price, "shock", EST_CFG.cum_horizon_quarters)
            price_rows.append({
                "place_id": pid,
                "place_name": g["place_name"].iloc[0],
                "nobs": int(m_price.nobs),
                "cum_1y": cum,
                "se_1y": se,
                "t_1y": t,
            })

        # PERMITS: use permits_g as dependent variable
        m_perm = fit_state_dlag(
            g, y_col="permits_g", x_base="shock",
            max_lag=EST_CFG.max_lag_quarters,
            hac_lags=EST_CFG.hac_lags
        )

        if m_perm is not None:
            cum, se, t = summarize_cum_effect(m_perm, "shock", EST_CFG.cum_horizon_quarters)
            perm_rows.append({
                "place_id": pid,
                "place_name": g["place_name"].iloc[0],
                "nobs": int(m_perm.nobs),
                "cum_1y": cum,
                "se_1y": se,
                "t_1y": t,
            })

    price = pd.DataFrame(price_rows).sort_values("cum_1y", ascending=True)
    perm = pd.DataFrame(perm_rows).sort_values("cum_1y", ascending=True)

    price.to_csv(STATE_PRICE_SENS_CSV, index=False)
    perm.to_csv(STATE_PERMITS_SENS_CSV, index=False)

    print(f"[SAVED] {STATE_PRICE_SENS_CSV} | rows={len(price)}")
    print(f"[SAVED] {STATE_PERMITS_SENS_CSV} | rows={len(perm)}")

    # quick peek: most negative / most positive
    print("\n[PRICE] most negative 5 (strongest drop):")
    print(price.head(5)[["place_id", "cum_1y", "t_1y"]].to_string(index=False))

    print("\n[PRICE] most positive 5:")
    print(price.tail(5)[["place_id", "cum_1y", "t_1y"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
