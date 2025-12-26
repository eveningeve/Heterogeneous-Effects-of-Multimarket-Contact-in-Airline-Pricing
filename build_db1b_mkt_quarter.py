# build_db1b_market_quarter.py
"""
DB1BMarket quarterly CSVs -> market-quarter price panel keyed like T-100.

Input:  DB1BMarket CSV files (with headers) in DB1B_RAW_DIR
        e.g. Origin_and_Destination_Survey_DB1BMarket_2010_1.csv

Output:
  - db1b_processed/parts/db1b_mkt_YYYY_Q#.parquet   (per-quarter)
  - db1b_processed/db1b_market_quarter.parquet      (stacked + re-aggregated)

Market key:
  - directionless city-market pair:
      mkt = min(OriginCityMarketID, DestCityMarketID) + "_" + max(...)
Time key:
  - qtr_id = Year*4 + Quarter

Aggregates (per market-quarter):
  - pax_db1b = sum(Passengers)
  - fare_mean = sum(MktFare * Passengers) / sum(Passengers)
  - n_rows = number of raw records contributing

Run:
  python build_db1b_market_quarter.py
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------
# CONFIG (edit these)
# -------------------------
DB1B_RAW_DIR = r"D:\MERGER IMPACT ANALYSIS\DB1B_raw"
OUT_DIR = r"D:\MERGER IMPACT ANALYSIS\db1b_processed"
CHUNKSIZE = 800_000  # adjust if memory tight

FINAL_NAME = "db1b_market_quarter.parquet"

# Must match your downloaded headers exactly
COL_YEAR = "Year"
COL_QUARTER = "Quarter"
COL_ORIG_CITY = "OriginCityMarketID"
COL_DEST_CITY = "DestCityMarketID"
COL_PAX = "Passengers"
COL_FARE = "MktFare"

# optional: keep for debugging/QA (set to True if you want)
KEEP_DEBUG_AIRPORTS = False
DEBUG_COLS = ["Origin", "Dest"]  # airport codes

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("db1b")

# -------------------------
# HELPERS
# -------------------------
def make_mkt_cityid(o: pd.Series, d: pd.Series) -> pd.Series:
    o = pd.to_numeric(o, errors="coerce")
    d = pd.to_numeric(d, errors="coerce")
    lo = np.minimum(o, d)
    hi = np.maximum(o, d)
    # keep as string key like T-100: "12345_67890"
    return lo.astype("Int64").astype(str) + "_" + hi.astype("Int64").astype(str)

def parse_yq_from_filename(name: str) -> tuple[int, int]:
    # expects ..._2010_1.csv or ..._2010_Q1.csv
    m = re.search(r"(20\d{2})[_\-]([1-4])(?=\.csv$)", name)
    if not m:
        # fallback: try any year + quarter in filename
        m = re.search(r"(20\d{2}).*?([1-4])", name)
    if not m:
        raise ValueError(f"Cannot parse year/quarter from filename: {name}")
    return int(m.group(1)), int(m.group(2))

def process_one_file(path: str, parts_dir: Path) -> Path:
    fp = Path(path)
    y, q = parse_yq_from_filename(fp.name)
    out_path = parts_dir / f"db1b_mkt_{y}_Q{q}.parquet"
    if out_path.exists():
        log.info(f"Skip (exists): {out_path.name}")
        return out_path

    log.info(f"Processing {fp.name}")

    usecols = [COL_YEAR, COL_QUARTER, COL_ORIG_CITY, COL_DEST_CITY, COL_PAX, COL_FARE]
    if KEEP_DEBUG_AIRPORTS:
        usecols += [c for c in DEBUG_COLS if c in pd.read_csv(fp, nrows=0).columns]

    # streaming aggregation:
    # store sums by (mkt, Year, Quarter)
    # We'll accumulate as dict: key -> (sum_pax, sum_wfare, n_rows)
    agg: dict[tuple[str, int, int], tuple[float, float, int]] = {}

    reader = pd.read_csv(fp, usecols=usecols, chunksize=CHUNKSIZE)

    for i, chunk in enumerate(reader, start=1):
        # coerce
        chunk[COL_YEAR] = pd.to_numeric(chunk[COL_YEAR], errors="coerce")
        chunk[COL_QUARTER] = pd.to_numeric(chunk[COL_QUARTER], errors="coerce")
        chunk[COL_ORIG_CITY] = pd.to_numeric(chunk[COL_ORIG_CITY], errors="coerce")
        chunk[COL_DEST_CITY] = pd.to_numeric(chunk[COL_DEST_CITY], errors="coerce")
        chunk[COL_PAX] = pd.to_numeric(chunk[COL_PAX], errors="coerce").fillna(0.0)
        chunk[COL_FARE] = pd.to_numeric(chunk[COL_FARE], errors="coerce")

        # drop rows missing keys or obviously invalid quarter/year
        chunk = chunk.dropna(subset=[COL_YEAR, COL_QUARTER, COL_ORIG_CITY, COL_DEST_CITY]).copy()
        chunk = chunk[(chunk[COL_YEAR] >= 1990) & (chunk[COL_YEAR] <= 2035)]
        chunk = chunk[(chunk[COL_QUARTER] >= 1) & (chunk[COL_QUARTER] <= 4)]

        # build keys
        chunk["mkt"] = make_mkt_cityid(chunk[COL_ORIG_CITY], chunk[COL_DEST_CITY])
        chunk["Year_i"] = chunk[COL_YEAR].astype(int)
        chunk["Quarter_i"] = chunk[COL_QUARTER].astype(int)

        # drop malformed markets (e.g., "<NA>_<NA>")
        chunk = chunk[~chunk["mkt"].str.contains("<NA>", na=False)]

        # weighted fare component
        chunk["wfare"] = chunk[COL_FARE] * chunk[COL_PAX]

        # group within chunk
        g = chunk.groupby(["mkt", "Year_i", "Quarter_i"], as_index=False).agg(
            sum_pax=(COL_PAX, "sum"),
            sum_wfare=("wfare", "sum"),
            n_rows=(COL_PAX, "size"),
        )

        # accumulate into dict
        for row in g.itertuples(index=False):
            key = (row.mkt, int(row.Year_i), int(row.Quarter_i))
            if key in agg:
                sp, sw, n = agg[key]
                agg[key] = (sp + float(row.sum_pax), sw + float(row.sum_wfare), n + int(row.n_rows))
            else:
                agg[key] = (float(row.sum_pax), float(row.sum_wfare), int(row.n_rows))

        if i % 3 == 0:
            log.info(f"  processed {i} chunk(s)")

    # finalize quarter df
    rows = []
    for (mkt, year, quarter), (sum_pax, sum_wfare, n_rows) in agg.items():
        fare_mean = np.nan if sum_pax <= 0 else (sum_wfare / sum_pax)
        qtr_id = year * 4 + quarter
        rows.append((mkt, year, quarter, qtr_id, sum_pax, fare_mean, n_rows))

    out = pd.DataFrame(rows, columns=["mkt", "YEAR", "quarter", "qtr_id", "pax_db1b", "fare_mean", "n_rows"])
    out = out.sort_values(["YEAR", "quarter", "mkt"]).reset_index(drop=True)

    parts_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info(f"  wrote {out_path} | rows={len(out):,} | mkts={out['mkt'].nunique():,}")
    return out_path


def main() -> None:
    raw_dir = Path(DB1B_RAW_DIR)
    out_dir = Path(OUT_DIR)
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    log.info(f"Found {len(files)} DB1BMarket CSV files in {raw_dir}")

    part_paths = [process_one_file(fp, parts_dir) for fp in files]

    log.info("Stacking parts into final market-quarter price panel...")

    all_df = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)

    # re-aggregate just in case
    final = (
        all_df.groupby(["mkt", "qtr_id", "YEAR", "quarter"], as_index=False)
              .agg(
                  pax_db1b=("pax_db1b", "sum"),
                  sum_wfare=("fare_mean", lambda x: np.nan),  # placeholder; we recompute below
                  n_rows=("n_rows", "sum"),
              )
    )

    # recompute fare_mean properly using pax weights:
    # easiest: merge back each part's fare_mean with its pax and recompute
    all_df["wfare"] = all_df["fare_mean"] * all_df["pax_db1b"]
    final2 = (
        all_df.groupby(["mkt", "qtr_id", "YEAR", "quarter"], as_index=False)
              .agg(
                  pax_db1b=("pax_db1b", "sum"),
                  wfare=("wfare", "sum"),
                  n_rows=("n_rows", "sum"),
              )
    )
    final2["fare_mean"] = np.where(final2["pax_db1b"] > 0, final2["wfare"] / final2["pax_db1b"], np.nan)
    final2 = final2.drop(columns=["wfare"]).sort_values(["YEAR","quarter","mkt"]).reset_index(drop=True)

    final_path = out_dir / FINAL_NAME
    final2.to_parquet(final_path, index=False)

    log.info(f"DONE. Final saved to: {final_path}")
    log.info(f"Rows: {len(final2):,} | mkts: {final2['mkt'].nunique():,} | years: {final2['YEAR'].min()}-{final2['YEAR'].max()}")

if __name__ == "__main__":
    main()
