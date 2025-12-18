# /src/11_summarize_robustness.py
from __future__ import annotations

import pandas as pd
from config import ensure_dirs, TABLE_DIR

IN_CSV  = TABLE_DIR / "robustness_grid.csv"
OUT_XLSX = TABLE_DIR / "robustness_summary.xlsx"
OUT_TXT  = TABLE_DIR / "robustness_snippet.txt"

def stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def main() -> int:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)

    # build readable coef cells
    df["saiz_cell"] = df.apply(
        lambda r: "" if pd.isna(r.get("b_saiz_elasticity")) else f"{r['b_saiz_elasticity']:.3f}{stars(r['p_saiz_elasticity'])}",
        axis=1
    )
    df["perm_cell"] = df.apply(
        lambda r: "" if pd.isna(r.get("b_permits_sens_1y")) else f"{r['b_permits_sens_1y']:.3f}{stars(r['p_permits_sens_1y'])}",
        axis=1
    )

    # filter to the multivariate model as primary robustness line
    mv = df[df["model"] == "price~saiz+permits"].copy()

    # Pivot: rows = shock_def, cols = K, values = coef
    saiz_piv = mv.pivot_table(index="shock_def", columns="K_lags", values="saiz_cell", aggfunc="first")
    perm_piv = mv.pivot_table(index="shock_def", columns="K_lags", values="perm_cell", aggfunc="first")

    # Also include N and R2
    n_piv  = mv.pivot_table(index="shock_def", columns="K_lags", values="nobs", aggfunc="first")
    r2_piv = mv.pivot_table(index="shock_def", columns="K_lags", values="r2", aggfunc="first")

    # Make pretty column names
    saiz_piv.columns = [f"K={c}" for c in saiz_piv.columns]
    perm_piv.columns = [f"K={c}" for c in perm_piv.columns]
    n_piv.columns    = [f"K={c}" for c in n_piv.columns]
    r2_piv.columns   = [f"K={c}" for c in r2_piv.columns]

    # Write Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        saiz_piv.reset_index().to_excel(writer, sheet_name="mv_saiz_coef", index=False)
        perm_piv.reset_index().to_excel(writer, sheet_name="mv_permits_coef", index=False)
        n_piv.reset_index().to_excel(writer, sheet_name="mv_N", index=False)
        r2_piv.reset_index().to_excel(writer, sheet_name="mv_R2", index=False)

        # full grid in case you need it
        df.to_excel(writer, sheet_name="full_grid", index=False)

    print(f"[SAVED] {OUT_XLSX}")

    # Snippet (generic template—numbers are in the table; snippet tells what to cite)
    snippet = []
    snippet.append("ROBUSTNESS SUMMARY (paste-ready)\n")
    snippet.append(
        "We assess robustness across alternative mortgage-rate shock definitions "
        "(deviation-from-trend with 8- and 12-quarter moving averages, quarterly changes, and year-over-year changes) "
        "and distributed-lag lengths (K=4,6,8). "
        "Across specifications, the sample remains 51 states (Saiz coverage: 42 states). "
        "Table robustness_summary.xlsx reports multivariate coefficients for Saiz elasticity and permits sensitivity.\n"
    )
    snippet.append(
        "Interpretation: if the sign/magnitude of the Saiz coefficient and permits coefficient remain broadly stable "
        "across shock definitions and lag lengths, the cross-state relationship is robust to these alternative choices.\n"
    )

    OUT_TXT.write_text("\n".join(snippet), encoding="utf-8")
    print(f"[SAVED] {OUT_TXT}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
