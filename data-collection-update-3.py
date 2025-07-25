#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_two_periods_flat.py

Load two periods of Bright Data JSON dumps (detailed + rank),
merge them, compute features (days_old, flags, discounts,
appearance_count, first_appearance), and save one row for
every ASIN appearance.
"""

import re, json
import pandas as pd
import numpy as np
from datetime import datetime

# ——— CONFIGURE ———
DETAILED1  = "/Users/leo/Downloads/bd_20250702_165527_0.json"
RANK1      = "/Users/leo/Downloads/bd_20250702_214241_0.json"
DETAILED2  = "/Users/leo/Downloads/bd_20250718_152158_0.json"
RANK2      = "/Users/leo/Downloads/bd_20250718_153434_0.json"
OUTPUT_CSV = "all_appearances_two_periods.csv"
# ——————————

def get_scrape_datetime(fname):
    m = re.search(r"(\d{8})_(\d{6})", fname)
    if not m:
        raise ValueError(f"Cannot parse datetime from '{fname}'")
    return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")

def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_amazon_data(jsfile):
    data = load_json_array(jsfile)
    rows = []
    for p in data:
        inp      = p.get("discovery_input") or {}
        details  = p.get("product_details") or []
        rows.append({
            "asin": p.get("asin"),
            "keyword": inp.get("keyword"),
            "is_prime": int(any("Prime" in d for d in (p.get("delivery") or []))),
            "brand": p.get("brand"),
            "rating": p.get("rating"),
            "reviews_count": p.get("reviews_count"),
            "initial_price": p.get("initial_price"),
            "final_price": p.get("final_price"),
            "availability": p.get("availability"),
            "bought_past_month": p.get("bought_past_month"),
            "bs_rank": p.get("bs_rank"),
            "bs_category": p.get("bs_category"),
            "amazon_choice": int(p.get("amazon_choice", False)),
            "seller_name": p.get("seller_name"),
            "ships_from": p.get("ships_from"),
            "plus_content": int(p.get("plus_content", False)),
            "colour": next((d["value"] for d in details if d.get("type")=="Color"), None),
            "material": next((d["value"] for d in details if d.get("type")=="Material"), None),
            "name": p.get("title"),
            "days_old_raw": next((d["value"] for d in details if d.get("type")=="Date First Available"), None)
        })
    return pd.DataFrame(rows)

def process_rank_data(jsfile):
    data = load_json_array(jsfile)
    rows = []
    for p in data:
        rows.append({
            "asin": p.get("asin"),
            "keyword_r": p.get("keyword"),
            "page_number": p.get("page_number"),
            "rank_on_page": p.get("rank_on_page"),
            "banner_product": int(p.get("is_banner_product", False)),
            "sponsored": int(str(p.get("sponsored","")).lower()=="true"),
            "name_r": p.get("name")
        })
    df = pd.DataFrame(rows)

    # compute overall search_rank
    pc = (
        df.groupby(["keyword_r","page_number"])
          .size()
          .to_frame("count")
          .reset_index()
    )
    pc["cum"] = pc.groupby("keyword_r")["count"].cumsum()
    pc["rank_offset"] = pc.groupby("keyword_r")["cum"].shift(1).fillna(0).astype(int)

    df = df.merge(
        pc[["keyword_r","page_number","rank_offset"]],
        on=["keyword_r","page_number"], how="left"
    )
    df["search_rank"] = df["rank_offset"] + df["rank_on_page"]

    return df.drop(columns=["rank_offset"])

def merge_period(detail_file, rank_file, time_val):
    scrape_dt = get_scrape_datetime(detail_file)

    df_d = process_amazon_data(detail_file)
    df_d["scrape_date"] = scrape_dt

    df_r = process_rank_data(rank_file)

    # 1) match on ASIN
    m1 = df_d.merge(df_r.dropna(subset=["asin"]), on="asin", how="inner")

    # 2) fallback: match remaining detailed rows on name_r
    used = set(m1["asin"])
    rem  = df_d[~df_d["asin"].isin(used)]
    m2 = rem.merge(
        df_r[df_r["asin"].isnull()],
        left_on="name",
        right_on="name_r",
        how="inner"
    )
    # In fallback merges the ASIN exists only on the detail side (asin_x) and is
    # missing on the rank side (asin_y).  Rename asin_x back to asin and drop asin_y.
    if "asin_x" in m2.columns:
        m2 = m2.rename(columns={"asin_x": "asin"})
        # Drop asin_y if present (it will be NaN)
        drop_cols = [c for c in ["asin_y"] if c in m2.columns]
        if drop_cols:
            m2 = m2.drop(columns=drop_cols)

    merged = pd.concat([m1, m2], ignore_index=True)
    merged["time"] = time_val

    # drop helper columns, including any leftover name/keyword from rank and
    # duplicate asin columns that may have survived merges
    for c in ("keyword_r", "name_r", "asin_x", "asin_y"):
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    return merged

def derive_features(df_all):
    # days_old
    def calc_days_old(raw, ref):
        if isinstance(raw, str):
            try:
                d = datetime.strptime(raw, "%B %d, %Y")
                return (ref - d).days
            except:
                return None
        return None

    df_all["days_old"] = df_all.apply(
        lambda r: calc_days_old(r["days_old_raw"], r["scrape_date"]), axis=1
    )

    # flags
    df_all["amazon_shipped"] = df_all["ships_from"].str.contains("Amazon", na=False).astype(int)
    df_all["full_stock"]     = df_all["availability"].str.contains("In Stock", na=False).astype(int)
    df_all["low_stock"]      = df_all["availability"].str.contains("Only", na=False).astype(int)
    df_all["no_stock"]       = df_all["availability"].str.contains("unavailable", na=False).astype(int)
    df_all["amazon_brand"]   = df_all["brand"].str.contains("Amazon", na=False).astype(int)

    # discounts
    df_all["discount_amount"] = df_all["initial_price"] - df_all["final_price"]
    discount_ratio = np.divide(
        df_all["discount_amount"],
        df_all["initial_price"],
        out=np.zeros(len(df_all), dtype=float),
        where=df_all["initial_price"].astype(float) != 0,
    )
    df_all["discount_percentage"] = pd.Series(discount_ratio, index=df_all.index).fillna(0) * 100

    # appearance_count
    ac = df_all.groupby(["asin","time"])["search_rank"].transform("count")
    df_all["appearance_count"] = ac.fillna(0).astype(int)

    # banner_count
    bc = df_all.groupby(["asin","time"])["banner_product"].transform("sum")
    df_all["banner_count"] = bc.fillna(0).astype(int)


    # first_appearance
    mins = df_all.groupby(["asin","time"])["search_rank"].transform("min")
    fa  = (df_all["search_rank"] == mins)
    df_all["first_appearance"] = fa.fillna(False).astype(int)

    # drop raw helpers
    df_all.drop(columns=["days_old_raw","scrape_date"], inplace=True)
    return df_all

if __name__ == "__main__":
    p1 = merge_period(DETAILED1, RANK1, time_val=1)
    p2 = merge_period(DETAILED2, RANK2, time_val=2)

    # stats derived from already loaded DataFrames
    unique_p1 = set(p1["asin"].dropna())
    unique_p2 = set(p2["asin"].dropna())

    print(
        f"Period 1 ASIN matches: {len(unique_p1)} "
        f"(detailed={len(unique_p1)}, rank={len(unique_p1)})"
    )
    print(
        f"Period 2 ASIN matches: {len(unique_p2)} "
        f"(detailed={len(unique_p2)}, rank={len(unique_p2)})"
    )

    both = unique_p1 & unique_p2
    print(f"ASINs present in both periods: {len(both)}")

    # --- stats for period 1 ---
    row_match1 = len(p1)
    unique_match1 = len(unique_p1)
    print(
        f"Period 1: row-level matches = {row_match1}, "
        f"unique ASIN matches = {unique_match1}"
    )

    # --- stats for period 2 ---
    row_match2 = len(p2)
    unique_match2 = len(unique_p2)
    print(
        f"Period 2: row-level matches = {row_match2}, "
        f"unique ASIN matches = {unique_match2}"
    )

    # --- overlap across periods (unique) ---
    both = set(p1['asin']) & set(p2['asin'])
    print(f"ASINs appearing in both periods (unique): {len(both)}")

    combined = pd.concat([p1, p2], ignore_index=True)
    final_df = derive_features(combined)
    final_df.sort_values(["time", "asin", "search_rank"], inplace=True)

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved all appearances to '{OUTPUT_CSV}' with {len(final_df)} rows.")