# build_t100_citypair_quarter.py
"""
BTS T-100 monthly .asc (no header) -> city-pair quarter-carrier parquet spine.

Input:  monthly .asc files in INPUT_DIR, e.g. 2010_Jan.asc, 2010_Feb.asc ...
Output:
  - processed/parts/*.parquet  (per-file aggregated chunks)
  - processed/t100_citypair_quarter.parquet (final stacked + re-aggregated)

Market definition:
  - Prefer NONSTOP city-pair using CITY_MARKET_IDs (directionless):
      mkt = min(origin_city_mkt, dest_city_mkt) + "_" + max(...)
  - Fallback to airport-pair (directionless) if city-market IDs cannot be detected.

Aggregates:
  - (mkt, YEAR, quarter, carrier): sum(seats, pax, dep)

Run:
  python build_t100_citypair_quarter.py
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = r"D:\MERGER IMPACT ANALYSIS\t100_raw"
OUTPUT_DIR = r"D:\MERGER IMPACT ANALYSIS\t100_processed"

FINAL_NAME = "t100_citypair_quarter.parquet"
CHUNKSIZE = 600_000

# Indices known to be stable in your extract (0-based)
IDX_YEAR = 0
IDX_MONTH = 1
IDX_PAX = 16
IDX_DEP = 17
IDX_SEATS = 26

SEP = "|"

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("t100")

# Helps detect carrier column
MAJOR_AIRLINES = {
    "AA","DL","UA","WN","US","AS","B6","F9","NK","HA","MQ","OO","9E","YV","YX","EV","XE","CO","NW","FL"
}

def _airline_score(series: pd.Series) -> float:
    s = series.astype(str).str.strip()
    return s.isin(MAJOR_AIRLINES).mean()

def _airport_share(series: pd.Series) -> float:
    s = series.astype(str).str.strip()
    return s.str.fullmatch(r"[A-Z]{3}").mean()

def _citymkt_share(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return ((x >= 10000) & (x <= 99999)).mean()

def month_to_quarter(month: pd.Series) -> pd.Series:
    return ((month - 1) // 3 + 1).astype("int8")

def make_mkt_citymkt(origin_city: pd.Series, dest_city: pd.Series) -> pd.Series:
    o = origin_city.astype("int64")
    d = dest_city.astype("int64")
    lo = o.where(o <= d, d)
    hi = d.where(o <= d, o)
    return lo.astype(str) + "_" + hi.astype(str)

def make_mkt_airport(o: pd.Series, d: pd.Series) -> pd.Series:
    o = o.astype(str).str.strip()
    d = d.astype(str).str.strip()
    lo = o.where(o <= d, d)
    hi = d.where(o <= d, o)
    return lo + "_" + hi

def validate_schema(path: str, sep: str="|", nrows: int=1000) -> None:
    """
    Lightweight sanity checks for YEAR/MONTH and numeric columns.
    """
    sample = pd.read_csv(path, sep=sep, header=None, engine="python", nrows=nrows)

    year_col = pd.to_numeric(sample[IDX_YEAR], errors="coerce")
    year_valid = ((year_col >= 1990) & (year_col <= 2030)).mean()
    if year_valid < 0.80:
        raise ValueError(f"Column {IDX_YEAR} doesn't look like YEAR (only {year_valid:.1%} in 1990-2030)")

    month_col = pd.to_numeric(sample[IDX_MONTH], errors="coerce")
    month_valid = ((month_col >= 1) & (month_col <= 12)).mean()
    if month_valid < 0.80:
        raise ValueError(f"Column {IDX_MONTH} doesn't look like MONTH (only {month_valid:.1%} in 1-12)")

    for idx, name in [(IDX_PAX, "PAX"), (IDX_DEP, "DEP"), (IDX_SEATS, "SEATS")]:
        col = pd.to_numeric(sample[idx], errors="coerce")
        numeric_share = (~col.isna()).mean()
        if numeric_share < 0.50:
            raise ValueError(f"Column {idx} ({name}) is only {numeric_share:.1%} numeric")
        col2 = col.dropna()
        if len(col2) and (col2 < 0).any():
            raise ValueError(f"Column {idx} ({name}) has negative values")

    log.info("  ✓ Schema validation passed")

def detect_schema(path: str, sep: str="|", nrows: int=60000):
    """
    Returns: carrier_idx, orig_idx, dest_idx, key_type
    key_type in {"citymkt","airport"}
    """
    sample = pd.read_csv(path, sep=sep, header=None, engine="python", nrows=nrows)

    # --- carrier column ---
    carrier_candidates = []
    for j in range(sample.shape[1]):
        col = sample[j]
        if _airport_share(col) > 0.80:
            continue
        score = _airline_score(col)
        if score > 0.005:
            carrier_candidates.append((j, score))

    if carrier_candidates:
        carrier_idx = max(carrier_candidates, key=lambda x: x[1])[0]
    else:
        def short_code_share(col):
            s = col.astype(str).str.strip()
            return s.str.fullmatch(r"[A-Z0-9]{2,3}").mean()

        non_airport = [j for j in range(sample.shape[1]) if _airport_share(sample[j]) < 0.80]
        if not non_airport:
            raise RuntimeError("Could not detect carrier column.")
        carrier_idx = max(non_airport, key=lambda j: short_code_share(sample[j]))

    # --- try city market IDs ---
    city_candidates = [(j, _citymkt_share(sample[j])) for j in range(sample.shape[1])]
    city_candidates = [c for c in city_candidates if c[1] > 0.30]
    city_candidates = sorted(city_candidates, key=lambda x: -x[1])[:12]
    city_idxs = [j for j, _ in city_candidates]

    def uniq_pairs_numeric(a_idx, b_idx) -> int:
        a = pd.to_numeric(sample[a_idx], errors="coerce")
        b = pd.to_numeric(sample[b_idx], errors="coerce")
        ok = (~a.isna()) & (~b.isna())
        if ok.mean() < 0.50:
            return 0
        a2 = a[ok].astype("int64").to_numpy()
        b2 = b[ok].astype("int64").to_numpy()
        lo = np.minimum(a2, b2)
        hi = np.maximum(a2, b2)
        return len(set(zip(lo, hi)))

    best_pair = None
    best_score = -1
    for i in range(len(city_idxs)):
        for k in range(i + 1, len(city_idxs)):
            a_idx, b_idx = city_idxs[i], city_idxs[k]
            score = uniq_pairs_numeric(a_idx, b_idx)
            if score > best_score:
                best_score = score
                best_pair = (a_idx, b_idx)

    if best_pair is not None and best_score >= 100:
        return carrier_idx, best_pair[0], best_pair[1], "citymkt"

    # --- fallback: airport codes ---
    airport_idxs = [j for j in range(sample.shape[1]) if _airport_share(sample[j]) > 0.80]

    def uniq_pairs_airport(a_idx, b_idx) -> int:
        a = sample[a_idx].astype(str).str.strip()
        b = sample[b_idx].astype(str).str.strip()
        ok = a.str.fullmatch(r"[A-Z]{3}") & b.str.fullmatch(r"[A-Z]{3}")
        if ok.mean() < 0.50:
            return 0
        a2 = a[ok].to_numpy()
        b2 = b[ok].to_numpy()
        lo = np.minimum(a2, b2)
        hi = np.maximum(a2, b2)
        return len(set(zip(lo, hi)))

    best_air = None
    best_air_score = -1
    for i in range(len(airport_idxs)):
        for k in range(i + 1, len(airport_idxs)):
            a_idx, b_idx = airport_idxs[i], airport_idxs[k]
            score = uniq_pairs_airport(a_idx, b_idx)
            if score > best_air_score:
                best_air_score = score
                best_air = (a_idx, b_idx)

    if best_air is not None and best_air_score >= 100:
        return carrier_idx, best_air[0], best_air[1], "airport"

    raise RuntimeError("Could not detect route key columns reliably.")

def process_one_file(path: str, parts_dir: Path) -> Path:
    log.info(f"Processing {Path(path).name}")

    validate_schema(path, sep=SEP, nrows=1000)

    carrier_idx, orig_idx, dest_idx, key_type = detect_schema(path, sep=SEP, nrows=60000)
    log.info(f"  detected idxs: carrier={carrier_idx}, orig={orig_idx}, dest={dest_idx}, key_type={key_type}")

    # IMPORTANT: usecols order must match names order (ascending not required; consistent order is)
    cols_map = {
        IDX_YEAR: "YEAR",
        IDX_MONTH: "MONTH",
        carrier_idx: "carrier",
        orig_idx: "orig",
        dest_idx: "dest",
        IDX_PAX: "pax",
        IDX_DEP: "dep",
        IDX_SEATS: "seats",
    }
    usecols = sorted(cols_map.keys())
    colnames = [cols_map[i] for i in usecols]

    reader = pd.read_csv(
        path,
        sep=SEP,
        header=None,
        usecols=usecols,
        names=colnames,
        chunksize=CHUNKSIZE,
        engine="python",
    )

    agg_chunks = []
    for i, df in enumerate(reader, start=1):
        # YEAR/MONTH always numeric
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").astype("Int64")

        # metrics numeric
        df["pax"] = pd.to_numeric(df["pax"], errors="coerce").fillna(0.0)
        df["dep"] = pd.to_numeric(df["dep"], errors="coerce").fillna(0.0)
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce").fillna(0.0)

        # drop missing basic keys (keep orig/dest as-is for now; key_type decides coercion)
        df = df.dropna(subset=["YEAR", "MONTH", "carrier", "orig", "dest"]).copy()

        df["quarter"] = month_to_quarter(df["MONTH"].astype("int64"))

        if key_type == "citymkt":
            # Only coerce orig/dest to numeric in citymkt mode
            orig_num = pd.to_numeric(df["orig"], errors="coerce")
            dest_num = pd.to_numeric(df["dest"], errors="coerce")

            # keep only values that are essentially integers (handles 33301.0 but drops 33301.1)
            orig_int = orig_num.where(orig_num.notna() & (orig_num % 1 == 0)).astype("Int64")
            dest_int = dest_num.where(dest_num.notna() & (dest_num % 1 == 0)).astype("Int64")

            df = df.assign(orig_int=orig_int, dest_int=dest_int).dropna(subset=["orig_int", "dest_int"]).copy()

            df["mkt"] = make_mkt_citymkt(df["orig_int"].astype("int64"), df["dest_int"].astype("int64"))

        else:
            # Keep orig/dest as strings in airport mode
            df["mkt"] = make_mkt_airport(df["orig"], df["dest"])

        g = (
            df.groupby(["mkt", "YEAR", "quarter", "carrier"], as_index=False)[["seats", "pax", "dep"]]
              .sum()
        )
        agg_chunks.append(g)

        if i % 5 == 0:
            log.info(f"  processed {i} chunk(s)")

    out_df = pd.concat(agg_chunks, ignore_index=True)
    out_df = (
        out_df.groupby(["mkt", "YEAR", "quarter", "carrier"], as_index=False)[["seats", "pax", "dep"]]
              .sum()
    )

    parts_dir.mkdir(parents=True, exist_ok=True)
    out_path = parts_dir / f"{Path(path).stem}.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info(f"  wrote {out_path} | rows={len(out_df):,} | mkts={out_df['mkt'].nunique():,}")
    return out_path

def main() -> None:
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    parts_dir = out_dir / "parts"

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(in_dir / "*.asc")))
    if not files:
        raise FileNotFoundError(f"No .asc files found in {in_dir}")

    log.info(f"Found {len(files)} .asc files in {in_dir}")

    # Strongly recommended: clear parts to avoid mixing old schema outputs
    parts_dir.mkdir(parents=True, exist_ok=True)
    for p in parts_dir.glob("*.parquet"):
        p.unlink()

    part_paths = []
    for fp in files:
        try:
            part_paths.append(process_one_file(fp, parts_dir))
        except Exception as e:
            log.exception(f"FAILED on {Path(fp).name}: {e}")
            raise

    log.info("Stacking parquet parts into final quarterly spine...")

    dfs = [pd.read_parquet(p) for p in part_paths]
    all_df = pd.concat(dfs, ignore_index=True)

    final_df = (
        all_df.groupby(["mkt", "YEAR", "quarter", "carrier"], as_index=False)[["seats", "pax", "dep"]]
              .sum()
              .sort_values(["YEAR", "quarter", "mkt", "carrier"])
              .reset_index(drop=True)
    )

    final_path = out_dir / FINAL_NAME
    final_df.to_parquet(final_path, index=False)

    log.info(f"DONE. Final saved to: {final_path}")
    log.info(
        f"Rows: {len(final_df):,} | markets: {final_df['mkt'].nunique():,} | carriers: {final_df['carrier'].nunique():,}"
    )

    pct_bad = ((final_df["dep"] == 0) & (final_df["pax"] > 0)).mean()
    log.info(f"Sanity check: share(dep==0 & pax>0) = {pct_bad:.4f}")

if __name__ == "__main__":
    main()
