# /src/07_viz_maps_and_scatter.py
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px

from config import (
    ensure_dirs,
    FIG_DIR,
    MAP_DIR,
    STATE_PRICE_SENS_CSV,
    STATE_PERMITS_SENS_CSV,
    SAIZ_STATE_CSV,
)

def main() -> int:
    ensure_dirs()

    price = pd.read_csv(STATE_PRICE_SENS_CSV).rename(columns={"cum_1y": "price_sens_1y"})
    perm = pd.read_csv(STATE_PERMITS_SENS_CSV).rename(columns={"cum_1y": "permits_sens_1y"})
    saiz = pd.read_csv(SAIZ_STATE_CSV)

    # Merge for scatters
    df = price.merge(perm[["place_id", "permits_sens_1y"]], on="place_id", how="left") \
              .merge(saiz, on="place_id", how="left")

    # -------------------------
    # 1) Choropleth maps (US states)
    # -------------------------
    # Price map
    fig_price = px.choropleth(
        price,
        locations="place_id",
        locationmode="USA-states",
        scope="usa",
        color="price_sens_1y",
        hover_name="place_name",
        hover_data={"place_id": True, "price_sens_1y": ":.3f", "t_1y": ":.2f"},
        title="State-level House Price Sensitivity (1-year cumulative effect)",
    )
    out_price = MAP_DIR / "map_price_sensitivity.html"
    fig_price.write_html(str(out_price), include_plotlyjs="cdn")
    print(f"[SAVED] {out_price}")

    # Permits map
    fig_perm = px.choropleth(
        perm,
        locations="place_id",
        locationmode="USA-states",
        scope="usa",
        color="permits_sens_1y",
        hover_name="place_name",
        hover_data={"place_id": True, "permits_sens_1y": ":.3f", "t_1y": ":.2f"},
        title="State-level Permits Sensitivity (1-year cumulative effect)",
    )
    out_perm = MAP_DIR / "map_permits_sensitivity.html"
    fig_perm.write_html(str(out_perm), include_plotlyjs="cdn")
    print(f"[SAVED] {out_perm}")

    # -------------------------
    # 2) Scatters
    # -------------------------
    # price vs permits
    fig_pp = px.scatter(
        df,
        x="permits_sens_1y",
        y="price_sens_1y",
        text="place_id",
        trendline="ols",
        hover_data={
            "place_id": True,
            "place_name": True,
            "price_sens_1y": ":.3f",
            "permits_sens_1y": ":.3f",
        },
        title="Price Sensitivity vs Permits Sensitivity",
    )
    fig_pp.update_traces(textposition="top center")
    out_pp = FIG_DIR / "scatter_price_vs_permits.html"
    fig_pp.write_html(str(out_pp), include_plotlyjs="cdn")
    print(f"[SAVED] {out_pp}")

    # price vs saiz (only states with saiz)
    df_saiz = df.dropna(subset=["saiz_elasticity"]).copy()
    fig_ps = px.scatter(
        df_saiz,
        x="saiz_elasticity",
        y="price_sens_1y",
        text="place_id",
        trendline="ols",
        hover_data={
            "place_id": True,
            "place_name": True,
            "price_sens_1y": ":.3f",
            "saiz_elasticity": ":.3f",
        },
        title="Price Sensitivity vs Saiz Housing Supply Elasticity (state aggregated)",
    )
    fig_ps.update_traces(textposition="top center")
    out_ps = FIG_DIR / "scatter_price_vs_saiz.html"
    fig_ps.write_html(str(out_ps), include_plotlyjs="cdn")
    print(f"[SAVED] {out_ps}")

    print("\n[DONE] Open the .html files in /data/processed/maps and /data/processed/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
