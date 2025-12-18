# /src/06_saiz_aggregate_and_regs.py
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    ensure_dirs,
    SAIZ_DTA,
    STATE_PRICE_SENS_CSV,
    STATE_PERMITS_SENS_CSV,
    SAIZ_STATE_CSV,
    REG_RESULTS_CSV,
)

# Census CBSA delineation (List 1, July 2023)
CENSUS_LIST1_URL = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"

# 2-digit state FIPS -> USPS
FIPS2USPS = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
    "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
    "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
    "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI",
    "56":"WY","72":"PR"
}


# -----------------------------
# Helpers
# -----------------------------
def _normalize_code(x) -> str:
    """Normalize numeric-like CBSA/MSA codes -> string without decimals."""
    if pd.isna(x):
        return ""
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def _norm_name(s: str) -> str:
    """
    Normalize metro names for matching:
      - uppercase
      - drop parentheses
      - remove common suffix noise
      - keep letters/numbers/space/comma/hyphen
    """
    s = str(s).upper()
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("&", " AND ")
    # remove some common geographic suffix noise
    for w in [
        " METROPOLITAN STATISTICAL AREA",
        " METROPOLITAN AREA",
        " MICRO AREA",
        " MICROPOLITAN AREA",
        " METRO AREA",
        " MSA",
        " CMSA",
        " NECMA",
        " NECTA",
    ]:
        s = s.replace(w, " ")
    # standardize separators
    s = s.replace("/", " ")
    s = re.sub(r"[^A-Z0-9\s,\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(name_norm: str) -> set:
    # tokens split on space/comma/hyphen; drop very short tokens
    parts = re.split(r"[ ,\-]+", name_norm)
    toks = {p for p in parts if len(p) >= 3 and p not in {"THE", "AND"}}
    return toks


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -----------------------------
# Census List1 parser
# -----------------------------
def _find_header_row(raw: pd.DataFrame) -> int:
    """
    Find the real header row in List1 Excel by scoring candidate rows with key strings.
    """
    keys = ["CBSA CODE", "CBSA TITLE", "FIPS STATE CODE", "FIPS COUNTY CODE"]
    best_row, best_score = None, -1
    scan = min(len(raw), 150)

    for i in range(scan):
        row = raw.iloc[i].astype(str).str.upper()
        score = sum(row.str.contains(k).any() for k in keys)
        if score > best_score:
            best_score = score
            best_row = i

    if best_row is None or best_score < 3:
        raise ValueError(f"List1 header row bulunamadı. best_row={best_row}, best_score={best_score}")

    return best_row


def build_cbsa_reference() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      - weights: CBSA -> (state, w) where w is county-count share inside CBSA
      - titles:  CBSA -> normalized title
    """
    raw = pd.read_excel(CENSUS_LIST1_URL, engine="openpyxl", header=None)
    header_row = _find_header_row(raw)

    x = pd.read_excel(CENSUS_LIST1_URL, engine="openpyxl", header=header_row)
    x.columns = [str(c).strip() for c in x.columns]
    cols_l = {c.lower(): c for c in x.columns}

    cbsa_col = next((orig for k, orig in cols_l.items() if "cbsa" in k and "code" in k), None)
    title_col = next((orig for k, orig in cols_l.items() if "cbsa" in k and "title" in k), None)
    stfips_col = next((orig for k, orig in cols_l.items() if "fips" in k and "state" in k and "code" in k), None)

    if cbsa_col is None or title_col is None or stfips_col is None:
        raise ValueError(
            "List1 gerekli kolonlar bulunamadı.\n"
            f"Kolonlar: {x.columns.tolist()}\n"
            f"cbsa_col={cbsa_col}, title_col={title_col}, stfips_col={stfips_col}"
        )

    df = x[[cbsa_col, title_col, stfips_col]].copy()
    df["cbsa"] = df[cbsa_col].apply(_normalize_code)

    df["stfips2"] = df[stfips_col].apply(_normalize_code).str.zfill(2)
    df["state"] = df["stfips2"].map(FIPS2USPS)

    df["title_norm"] = df[title_col].apply(_norm_name)

    df = df.dropna(subset=["cbsa", "state", "title_norm"])
    df = df[df["cbsa"] != ""]

    # weights: list1 is county-level, so each row is a county in a CBSA
    w = df.groupby(["cbsa", "state"]).size().reset_index(name="n_counties")
    w["w"] = w["n_counties"] / w.groupby("cbsa")["n_counties"].transform("sum")
    w = w[["cbsa", "state", "w"]]

    titles = df[["cbsa", "title_norm"]].drop_duplicates()

    return w, titles


# -----------------------------
# Matching
# -----------------------------
def match_saiz_to_cbsa(saiz_names: pd.Series, titles: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      msaname_norm, cbsa, match_score
    """
    title_norms = titles["title_norm"].tolist()
    cbsa_by_title = dict(zip(titles["title_norm"], titles["cbsa"]))

    # Precompute tokens for speed
    title_tokens = {t: _tokenize(t) for t in title_norms}

    def best_match(name_norm: str) -> Tuple[str | None, float]:
        if not name_norm:
            return None, 0.0

        # exact
        if name_norm in cbsa_by_title:
            return cbsa_by_title[name_norm], 1.0

        toks = _tokenize(name_norm)

        # Stage 1: get top candidates by Jaccard token overlap
        # (avoid scanning all titles with SequenceMatcher)
        scored = []
        for t in title_norms:
            j = _jaccard(toks, title_tokens[t])
            if j >= 0.20:  # loose prefilter
                scored.append((j, t))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [t for _, t in scored[:50]] if scored else title_norms[:80]

        # Stage 2: SequenceMatcher on top candidates
        best_t, best_sc = None, 0.0
        for t in top:
            sc = SequenceMatcher(None, name_norm, t).ratio()
            # small bonus for substring
            if name_norm in t or t in name_norm:
                sc += 0.03
            if sc > best_sc:
                best_sc, best_t = sc, t

        if best_t is None:
            return None, 0.0
        return cbsa_by_title[best_t], float(best_sc)

    out = []
    for nm in saiz_names.dropna().unique().tolist():
        cbsa, sc = best_match(nm)
        if cbsa is not None and sc >= min_score:
            out.append({"msaname_norm": nm, "cbsa": cbsa, "match_score": sc})

    return pd.DataFrame(out)


# -----------------------------
# Regressions
# -----------------------------
def ols_table(df: pd.DataFrame, y: str, X: List[str], name: str) -> pd.DataFrame:
    d = df.dropna(subset=[y] + X).copy()
    if len(d) < 10:
        # return empty if too few obs
        return pd.DataFrame([{
            "model": name, "var": "N/A", "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
            "nobs": int(len(d)), "r2": np.nan
        }])

    Xmat = sm.add_constant(d[X])
    model = sm.OLS(d[y], Xmat).fit(cov_type="HC1")  # robust SE

    rows = []
    for v in ["const"] + X:
        rows.append({
            "model": name,
            "var": v,
            "coef": float(model.params.get(v)),
            "se": float(model.bse.get(v)),
            "t": float(model.tvalues.get(v)),
            "p": float(model.pvalues.get(v)),
            "nobs": int(model.nobs),
            "r2": float(model.rsquared),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ensure_dirs()

    # 1) Load state sensitivities
    price = pd.read_csv(STATE_PRICE_SENS_CSV).rename(columns={"cum_1y": "price_sens_1y"})
    perm = pd.read_csv(STATE_PERMITS_SENS_CSV).rename(columns={"cum_1y": "permits_sens_1y"})

    # 2) Load Saiz
    saiz = pd.read_stata(SAIZ_DTA)
    needed = {"msaname", "elasticity"}
    if not needed.issubset(set(saiz.columns)):
        raise ValueError(f"Saiz kolonları beklenmedik. Kolonlar: {saiz.columns.tolist()}")

    saiz_sub = saiz[["msaname", "elasticity"]].copy()
    saiz_sub["msaname_norm"] = saiz_sub["msaname"].apply(_norm_name)
    saiz_sub = saiz_sub.dropna(subset=["elasticity", "msaname_norm"])
    saiz_sub = saiz_sub[saiz_sub["msaname_norm"] != ""].copy()

    # 3) Build CBSA reference (weights + titles) from Census (auto download)
    w, titles = build_cbsa_reference()

    # 4) Match Saiz MSA names to CBSA titles (adaptive cutoff)
    cutoffs = [0.84, 0.82, 0.80, 0.78]
    match_df = None
    for c in cutoffs:
        tmp = match_saiz_to_cbsa(saiz_sub["msaname_norm"], titles, min_score=c)
        # We want at least some decent coverage (e.g., >= 50 matched MSAs)
        if len(tmp) >= 50 or c == cutoffs[-1]:
            match_df = tmp
            print(f"[MATCH] cutoff={c:.2f} | matched_unique_msas={len(tmp)}")
            break

    if match_df is None or match_df.empty:
        raise RuntimeError("Saiz->CBSA eşleşmesi 0 geldi. Normalizasyonu genişletmek gerekir.")

    # 5) Attach CBSA to Saiz rows
    saiz_m = saiz_sub.merge(match_df[["msaname_norm", "cbsa", "match_score"]], on="msaname_norm", how="inner")

    print(f"[SAIZ] matched rows={len(saiz_m)} | avg_score={saiz_m['match_score'].mean():.3f}")

    # 6) Merge with CBSA->state weights, then aggregate to state
    m = saiz_m.merge(w, on="cbsa", how="left")
    miss_state = m["state"].isna().mean()
    if miss_state > 0:
        print(f"[WARN] matched CBSA'ların %{miss_state*100:.1f} için state bulunamadı (beklenmez).")

    m = m.dropna(subset=["state", "elasticity", "w"]).copy()
    m["wx"] = m["w"] * m["elasticity"]

    # State elasticity = weighted avg across (CBSA within state), weights are county-share inside each CBSA.
    saiz_state = (
        m.groupby("state", as_index=False)
         .agg(sum_w=("w", "sum"), sum_wx=("wx", "sum"))
    )
    saiz_state["saiz_elasticity"] = saiz_state["sum_wx"] / saiz_state["sum_w"]
    saiz_state = saiz_state.rename(columns={"state": "place_id"})[["place_id", "saiz_elasticity"]]

    if saiz_state.empty:
        raise RuntimeError("Saiz state agregasyon boş kaldı. (Eşleşme/weights sorunu)")

    saiz_state.to_csv(SAIZ_STATE_CSV, index=False)
    print(f"[SAVED] {SAIZ_STATE_CSV} | rows={len(saiz_state)}")

    # 7) Merge all state-level pieces
    df = (
        price.merge(perm[["place_id", "permits_sens_1y"]], on="place_id", how="left")
             .merge(saiz_state, on="place_id", how="left")
    )

    # 8) Regressions
    reg = pd.concat([
        ols_table(df, y="price_sens_1y", X=["saiz_elasticity"], name="(1) price~saiz"),
        ols_table(df, y="price_sens_1y", X=["permits_sens_1y"], name="(2) price~permits"),
        ols_table(df, y="price_sens_1y", X=["saiz_elasticity", "permits_sens_1y"], name="(3) price~saiz+permits"),
    ], ignore_index=True)

    reg.to_csv(REG_RESULTS_CSV, index=False)
    print(f"[SAVED] {REG_RESULTS_CSV}")

    print("\n[REG] preview:")
    print(reg.head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
