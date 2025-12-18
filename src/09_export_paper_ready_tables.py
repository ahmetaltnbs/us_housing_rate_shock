# /src/09_export_paper_ready_tables.py
from __future__ import annotations

import math
from pathlib import Path
import pandas as pd

from config import (
    ensure_dirs,
    TABLE_DIR,
    STATE_PRICE_SENS_CSV,
    STATE_PERMITS_SENS_CSV,
    SAIZ_STATE_CSV,
    REG_RESULTS_CSV,
)

OUT_XLSX = TABLE_DIR / "paper_ready_tables.xlsx"
OUT_MD   = TABLE_DIR / "paper_ready_tables.md"
OUT_TEX  = TABLE_DIR / "paper_ready_tables.tex"
OUT_TXT  = TABLE_DIR / "results_snippets.txt"


# -------------------------
# Helpers
# -------------------------
def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_from_t_norm(t: float) -> float:
    """Two-sided p-value using normal approx."""
    if pd.isna(t):
        return float("nan")
    z = abs(float(t))
    return 2.0 * (1.0 - norm_cdf(z))


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt(x, nd=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def make_top_bottom(df: pd.DataFrame, col: str, n: int = 10):
    d = df.sort_values(col, ascending=True).copy()
    bottom = d.head(n).copy()
    top = d.tail(n).sort_values(col, ascending=False).copy()
    return top, bottom


def to_markdown_table(df: pd.DataFrame, title: str) -> str:
    # tabulate bağımlılığı olmadan markdown table üretir
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")

    return "\n### " + title + "\n\n" + "\n".join([header, sep] + rows) + "\n"



def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    rep = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for k, v in rep.items():
        s = s.replace(k, v)
    return s


def df_to_latex_tabular(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = [latex_escape(c) for c in df.columns]
    colspec = "l" * len(cols)
    header = " & ".join(cols) + r" \\"
    lines = [r"\begin{table}[!htbp]",
             r"\centering",
             rf"\caption{{{latex_escape(caption)}}}",
             rf"\label{{{latex_escape(label)}}}",
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\hline",
             header,
             r"\hline"]
    for _, row in df.iterrows():
        vals = [latex_escape(row[c]) if isinstance(row[c], str) else latex_escape(str(row[c])) for c in df.columns]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# -------------------------
# Main
# -------------------------
def main() -> int:
    ensure_dirs()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    price = pd.read_csv(STATE_PRICE_SENS_CSV).rename(columns={"cum_1y": "price_sens_1y"})
    perm  = pd.read_csv(STATE_PERMITS_SENS_CSV).rename(columns={"cum_1y": "permits_sens_1y"})
    saiz  = pd.read_csv(SAIZ_STATE_CSV)
    reg   = pd.read_csv(REG_RESULTS_CSV)

    # ---- Create p-values + stars for sensitivities (normal approx)
    for df, tcol, pcol, scol in [
        (price, "t_1y", "p_approx", "sig"),
        (perm,  "t_1y", "p_approx", "sig"),
    ]:
        df[pcol] = df[tcol].apply(p_from_t_norm)
        df[scol] = df[pcol].apply(stars)

    # ---- Table 1: Price sensitivity top/bottom 10
    p_top, p_bottom = make_top_bottom(price, "price_sens_1y", 10)

    def prep_sens_table(d: pd.DataFrame, sens_col: str) -> pd.DataFrame:
        out = d[["place_id", "place_name", sens_col, "se_1y", "t_1y", "p_approx", "sig"]].copy()
        out["effect"] = out[sens_col].map(lambda x: fmt(x, 3))
        out["se"] = out["se_1y"].map(lambda x: fmt(x, 3))
        out["t"] = out["t_1y"].map(lambda x: fmt(x, 2))
        out["p"] = out["p_approx"].map(lambda x: fmt(x, 3))
        out["sig"] = out["sig"].astype(str)
        out = out.rename(columns={"place_id": "State", "place_name": "State name"})
        return out[["State", "State name", "effect", "se", "t", "p", "sig"]]

    t1_bottom = prep_sens_table(p_bottom, "price_sens_1y")
    t1_top    = prep_sens_table(p_top, "price_sens_1y")

    # ---- Table 2: Permits sensitivity top/bottom 10
    pr_top, pr_bottom = make_top_bottom(perm, "permits_sens_1y", 10)
    t2_bottom = prep_sens_table(pr_bottom, "permits_sens_1y")
    t2_top    = prep_sens_table(pr_top, "permits_sens_1y")

    # ---- Table 3: Cross-state regressions (clean wide format)
    # reg has rows: model,var,coef,se,t,p,nobs,r2
    # Build a wide table: variables as rows, models as columns with coef (se) + stars.
    reg2 = reg.copy()
    reg2["sig"] = reg2["p"].apply(stars)
    reg2["cell"] = reg2.apply(lambda r: f"{r['coef']:.3f}{r['sig']} ({r['se']:.3f})" if pd.notna(r["coef"]) else "", axis=1)
    models = reg2["model"].unique().tolist()

    # keep only key vars in nice order
    var_order = ["saiz_elasticity", "permits_sens_1y", "const"]
    nice_name = {
        "saiz_elasticity": "Saiz elasticity",
        "permits_sens_1y": "Permits sensitivity (1y)",
        "const": "Constant",
    }

    rows = []
    for v in var_order:
        row = {"Variable": nice_name.get(v, v)}
        for m in models:
            cell = reg2.loc[(reg2["model"] == m) & (reg2["var"] == v), "cell"]
            row[m] = cell.iloc[0] if len(cell) else ""
        rows.append(row)

    t3 = pd.DataFrame(rows)

    # Add N and R2 as separate rows
    n_row = {"Variable": "N"}
    r2_row = {"Variable": "R²"}
    for m in models:
        sub = reg2[reg2["model"] == m]
        n_row[m] = str(int(sub["nobs"].dropna().iloc[0])) if sub["nobs"].notna().any() else ""
        r2_row[m] = f"{float(sub['r2'].dropna().iloc[0]):.3f}" if sub["r2"].notna().any() else ""
    t3 = pd.concat([t3, pd.DataFrame([n_row, r2_row])], ignore_index=True)

    # ---- Saiz coverage note
    n_states_total = int(price["place_id"].nunique())
    n_saiz_states  = int(saiz["place_id"].nunique())

    # ---- Results snippets
    # strongest negative/positive + significant ones
    most_neg = price.sort_values("price_sens_1y").head(5)[["place_id", "price_sens_1y", "t_1y"]]
    most_pos = price.sort_values("price_sens_1y", ascending=False).head(5)[["place_id", "price_sens_1y", "t_1y"]]

    # pull key regression coefficients
    def get_coef(model_name: str, var: str):
        s = reg2[(reg2["model"] == model_name) & (reg2["var"] == var)]
        if len(s) == 0:
            return None
        return float(s["coef"].iloc[0]), float(s["se"].iloc[0]), float(s["p"].iloc[0])

    m1 = models[0]  # (1) price~saiz
    m3 = models[-1] # (3) price~saiz+permits

    c_saiz_m1 = get_coef(m1, "saiz_elasticity")
    c_saiz_m3 = get_coef(m3, "saiz_elasticity")
    c_perm_m3 = get_coef(m3, "permits_sens_1y")

    lines = []
    lines.append("RESULTS SNIPPETS (paste-ready)\n")
    lines.append(f"Sample: {n_states_total} states (incl. DC). Saiz elasticity coverage: {n_saiz_states} states.\n")

    # paragraph 1: heterogeneity
    lines.append(
        "Cross-state heterogeneity in the 1-year cumulative house price response to a mortgage-rate shock is sizeable. "
        f"The most negative responses are observed in {', '.join(most_neg['place_id'].tolist())}, "
        "while the most positive responses occur in "
        f"{', '.join(most_pos['place_id'].tolist())}.\n"
    )

    # paragraph 2: regression summary
    if c_saiz_m1:
        b, se, p = c_saiz_m1
        lines.append(
            f"In cross-state regressions, the bivariate association between price sensitivity and Saiz elasticity "
            f"is positive (coef={b:.3f}, SE={se:.3f}, p={p:.3f}; N={n_saiz_states}). "
        )
    if c_saiz_m3 and c_perm_m3:
        b1, se1, p1 = c_saiz_m3
        b2, se2, p2 = c_perm_m3
        lines.append(
            f"Controlling jointly for Saiz elasticity and permits sensitivity, both coefficients remain positive "
            f"(Saiz: {b1:.3f}, SE={se1:.3f}, p={p1:.3f}; "
            f"Permits: {b2:.3f}, SE={se2:.3f}, p={p2:.3f}).\n"
        )

    lines.append(
        "Note: p-values shown in the sensitivity ranking tables are normal-approximation p-values based on the reported t-statistics "
        "(HAC/Newey-West standard errors were used in the state time-series regressions).\n"
    )

    # ---- Export Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        t1_bottom.to_excel(writer, sheet_name="T1_price_bottom10", index=False)
        t1_top.to_excel(writer, sheet_name="T1_price_top10", index=False)
        t2_bottom.to_excel(writer, sheet_name="T2_permits_bottom10", index=False)
        t2_top.to_excel(writer, sheet_name="T2_permits_top10", index=False)
        t3.to_excel(writer, sheet_name="T3_cross_state_regs", index=False)

        # Notes sheet
        notes = pd.DataFrame({
            "note": [
                "Table T1/T2: p-values are normal-approximation based on t-statistics (two-sided). Stars: * p<0.10, ** p<0.05, *** p<0.01.",
                f"Saiz state coverage: {n_saiz_states} out of {n_states_total} states.",
                "Table T3: robust (HC1) standard errors. Cell format: coef(stars) (se).",
            ]
        })
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"[SAVED] {OUT_XLSX}")

    # ---- Export Markdown
    md = []
    md.append("# Paper-ready Tables\n")
    md.append(f"- States total: **{n_states_total}** (incl. DC)\n- Saiz-covered states: **{n_saiz_states}**\n")
    md.append(to_markdown_table(t1_bottom, "Table 1A: Price sensitivity (Bottom 10; most negative)"))
    md.append(to_markdown_table(t1_top,    "Table 1B: Price sensitivity (Top 10; most positive)"))
    md.append(to_markdown_table(t2_bottom, "Table 2A: Permits sensitivity (Bottom 10)"))
    md.append(to_markdown_table(t2_top,    "Table 2B: Permits sensitivity (Top 10)"))
    md.append(to_markdown_table(t3,        "Table 3: Cross-state regressions (coef (se), robust SE)"))
    md.append("\n**Notes:** Stars: * p<0.10, ** p<0.05, *** p<0.01. "
              "T1/T2 p-values are normal-approx. T3 uses robust (HC1) SE.\n")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[SAVED] {OUT_MD}")

    # ---- Export LaTeX
    tex = []
    tex.append(r"% Paper-ready LaTeX tables" + "\n")
    tex.append(df_to_latex_tabular(t1_bottom, "Price sensitivity (Bottom 10; most negative)", "tab:price_bottom10"))
    tex.append(df_to_latex_tabular(t1_top,    "Price sensitivity (Top 10; most positive)", "tab:price_top10"))
    tex.append(df_to_latex_tabular(t2_bottom, "Permits sensitivity (Bottom 10)", "tab:permits_bottom10"))
    tex.append(df_to_latex_tabular(t2_top,    "Permits sensitivity (Top 10)", "tab:permits_top10"))
    tex.append(df_to_latex_tabular(t3,        "Cross-state regressions (coef (se), robust SE)", "tab:cross_state_regs"))
    OUT_TEX.write_text("\n".join(tex), encoding="utf-8")
    print(f"[SAVED] {OUT_TEX}")

    # ---- Export Results snippet text
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {OUT_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
