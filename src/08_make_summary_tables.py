# /src/08_make_summary_tables.py
from __future__ import annotations

import pandas as pd

from config import (
    ensure_dirs,
    TABLE_DIR,
    STATE_PRICE_SENS_CSV,
    STATE_PERMITS_SENS_CSV,
    SAIZ_STATE_CSV,
)

OUT_XLSX = TABLE_DIR / "summary_tables.xlsx"


def top_bottom(df: pd.DataFrame, col: str, n: int = 10):
    df2 = df.sort_values(col, ascending=True).copy()
    bottom = df2.head(n)
    top = df2.tail(n).sort_values(col, ascending=False)
    return top, bottom


def main() -> int:
    ensure_dirs()

    price = pd.read_csv(STATE_PRICE_SENS_CSV).rename(columns={"cum_1y": "price_sens_1y"})
    perm = pd.read_csv(STATE_PERMITS_SENS_CSV).rename(columns={"cum_1y": "permits_sens_1y"})
    saiz = pd.read_csv(SAIZ_STATE_CSV)

    # --- Price ranking
    p_top, p_bottom = top_bottom(price, "price_sens_1y", 10)

    # --- Permits ranking
    pr_top, pr_bottom = top_bottom(perm, "permits_sens_1y", 10)

    # --- Saiz coverage + distribution
    saiz_states = saiz["place_id"].nunique()
    price_saiz = price.merge(saiz, on="place_id", how="inner")
    ps_top, ps_bottom = top_bottom(price_saiz, "price_sens_1y", 10)

    # --- Basic descriptives
    desc_price = price[["price_sens_1y", "se_1y", "t_1y"]].describe()
    desc_perm = perm[["permits_sens_1y", "se_1y", "t_1y"]].describe()
    desc_saiz = saiz[["saiz_elasticity"]].describe()

    # --- Save to Excel (multiple sheets)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        p_top.to_excel(writer, sheet_name="price_top10", index=False)
        p_bottom.to_excel(writer, sheet_name="price_bottom10", index=False)

        pr_top.to_excel(writer, sheet_name="permits_top10", index=False)
        pr_bottom.to_excel(writer, sheet_name="permits_bottom10", index=False)

        ps_top.to_excel(writer, sheet_name="price_top10_saiz", index=False)
        ps_bottom.to_excel(writer, sheet_name="price_bottom10_saiz", index=False)

        desc_price.to_excel(writer, sheet_name="desc_price")
        desc_perm.to_excel(writer, sheet_name="desc_permits")
        desc_saiz.to_excel(writer, sheet_name="desc_saiz")

        pd.DataFrame({"saiz_states_covered": [saiz_states]}).to_excel(writer, sheet_name="saiz_coverage", index=False)

    print(f"[SAVED] {OUT_XLSX}")
    print(f"[SAIZ] states covered: {saiz_states}")

    # quick console preview
    print("\n[PRICE] Bottom 10 (most negative):")
    print(p_bottom[["place_id", "price_sens_1y", "t_1y"]].to_string(index=False))

    print("\n[PRICE] Top 10 (most positive):")
    print(p_top[["place_id", "price_sens_1y", "t_1y"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
