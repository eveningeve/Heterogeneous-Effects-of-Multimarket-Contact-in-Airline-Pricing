# build_t100_mktq_citymarket_fixed.py
"""
T-100 monthly .asc (no header) -> market-quarter structure panel (CityMarketID endpoints).

KEYS (fixed, verified):
  YEAR  = col 0
  MONTH = col 1
  ORIG_CITY_MKT = col 3
  DEST_CITY_MKT = col 7
  CARRIER = col 10
  PAX   = col 16
  DEP   = col 17
  SEATS = col 26   (use your cleaned file if seats issues already handled)

Outputs:
  - t100_mktq_citymarket.parquet : one row per (mkt, qtr_id)
    with seats_total, pax_total, dep_total, n_carriers, HHI_seats
"""

from __future__ import annotations
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
INPUT_DIR  = r"D:\MERGER IMPACT ANALYSIS\t100_raw"
OUTPUT_DIR = r"D:\MERGER IMPACT ANALYSIS\t100_processed"
FINAL_NAME = "t100_mktq_citymarket.parquet"
CHUNKSIZE  = 600_000
SEP = "|"

IDX_YEAR = 0
IDX_MONTH = 1
IDX_ORIG_CITY_MKT = 3
IDX_DEST_CITY_MKT = 7
IDX_CARRIER = 10
IDX_PAX = 16
IDX_DEP = 17
IDX_SEATS = 26

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("t100_mktq")

def month_to_quarter(m: pd.Series) -> pd.Series:
    return ((m - 1) // 3 + 1).astype("int8")

def make_mkt(o: pd.Series, d: pd.Series) -> pd.Series:
    o = o.astype("int64")
    d = d.astype("int64")
    lo = o.where(o <= d, d)
    hi = d.where(o <= d, o)
    return lo.astype(str) + "_" + hi.astype(str)

def process_one_file(path: str) -> pd.DataFrame:
    fp = Path(path)
    log.info(f"Processing {fp.name}")

    usecols = [IDX_YEAR, IDX_MONTH, IDX_ORIG_CITY_MKT, IDX_DEST_CITY_MKT,
               IDX_CARRIER, IDX_PAX, IDX_DEP, IDX_SEATS]
    names = ["YEAR","MONTH","orig_city_mkt","dest_city_mkt","carrier","pax","dep","seats"]

    reader = pd.read_csv(
        path, sep=SEP, header=None, usecols=usecols, names=names,
        chunksize=CHUNKSIZE, engine="python"
    )

    out_chunks = []
    for df in reader:
        # numeric coercion
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        df["orig_city_mkt"] = pd.to_numeric(df["orig_city_mkt"], errors="coerce")
        df["dest_city_mkt"] = pd.to_numeric(df["dest_city_mkt"], errors="coerce")

        df["pax"] = pd.to_numeric(df["pax"], errors="coerce").fillna(0.0)
        df["dep"] = pd.to_numeric(df["dep"], errors="coerce").fillna(0.0)
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce").fillna(0.0)

        # drop missing keys
        df = df.dropna(subset=["YEAR","MONTH","orig_city_mkt","dest_city_mkt","carrier"]).copy()

        # cast ints safely
        df["YEAR"] = df["YEAR"].astype("int64")
        df["MONTH"] = df["MONTH"].astype("int64")
        df["orig_city_mkt"] = df["orig_city_mkt"].astype("int64")
        df["dest_city_mkt"] = df["dest_city_mkt"].astype("int64")

        # quarter + qtr_id
        df["quarter"] = month_to_quarter(df["MONTH"])
        df["qtr_id"] = df["YEAR"] * 4 + df["quarter"].astype("int64")

        # market key
        df["mkt"] = make_mkt(df["orig_city_mkt"], df["dest_city_mkt"])

        # carrier-level aggregation within month-file chunk
        g = df.groupby(["mkt","qtr_id","carrier"], as_index=False)[["seats","pax","dep"]].sum()
        out_chunks.append(g)

    return pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame(
        columns=["mkt","qtr_id","carrier","seats","pax","dep"]
    )

def main():
    files = sorted(glob.glob(str(Path(INPUT_DIR) / "*.asc")))
    if not files:
        raise FileNotFoundError(f"No .asc files found in {INPUT_DIR}")
    log.info(f"Found {len(files)} files")

    # 1) Build carrier-level quarterly contributions
    carrier_parts = []
    for fp in files:
        carrier_parts.append(process_one_file(fp))
    carrier = pd.concat(carrier_parts, ignore_index=True)

    # collapse across files (same qtr appears in 3 months)
    carrier = carrier.groupby(["mkt","qtr_id","carrier"], as_index=False)[["seats","pax","dep"]].sum()
    carrier_out = Path(OUTPUT_DIR) / "t100_carrierq_citymarket.parquet"
    carrier.to_parquet(carrier_out, index=False)

    log.info(
    f"Saved carrier-level file | rows={len(carrier):,} "
    f"| mkts={carrier['mkt'].nunique():,} "
    f"| carriers={carrier['carrier'].nunique():,}"
)

    # 2) Market-quarter totals
    mq = carrier.groupby(["mkt","qtr_id"], as_index=False).agg(
        seats_total=("seats","sum"),
        pax_total=("pax","sum"),
        dep_total=("dep","sum"),
        n_carriers=("carrier","nunique"),
    )

    # 3) HHI by seats
    tmp = carrier.merge(mq[["mkt","qtr_id","seats_total"]], on=["mkt","qtr_id"], how="left", validate="many_to_one")
    tmp["share_seats"] = np.where(tmp["seats_total"]>0, tmp["seats"]/tmp["seats_total"], 0.0)

    hhi = tmp.groupby(["mkt","qtr_id"], as_index=False).agg(
        HHI_seats=("share_seats", lambda s: float(np.sum(np.square(s))))
    )

    t100_mkt = mq.merge(hhi, on=["mkt","qtr_id"], how="left", validate="one_to_one")

    # derive YEAR/quarter
    t100_mkt["YEAR"] = ((t100_mkt["qtr_id"] - 1)//4).astype("int64")
    t100_mkt["quarter"] = (((t100_mkt["qtr_id"] - 1)%4)+1).astype("int64")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / FINAL_NAME
    t100_mkt.to_parquet(out_path, index=False)

    log.info(f"Saved: {out_path}")
    log.info(f"Rows: {len(t100_mkt):,} | mkts: {t100_mkt['mkt'].nunique():,}")

if __name__ == "__main__":
    main()