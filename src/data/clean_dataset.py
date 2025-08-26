import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import pycountry

RAW_PATH = os.environ.get("RAW_PATH", "data/raw/world_economics.csv")
OUT_DIR = os.environ.get("OUT_DIR", "data/processed")
OUT_CSV = os.path.join(OUT_DIR, "world_economics_clean.csv")
OUT_PARQUET = os.path.join(OUT_DIR, "world_economics_clean.parquet")

NUMERIC_COLS = [
    "GDP",
    "GDP Growth",
    "Interest Rate",
    "Inflation Rate",
    "Jobless Rate",
    "Gov. Budget",
    "Debt/GDP",
    "Current Account",
    "Population",
]

CATEGORY_COLS = ["name", "region", "subregion", "currency", "capital", "languages"]

ISO_ALIASES: Dict[str, str] = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Ivory Coast": "CIV",
    "Congo": "COG",
    "Republic of the Congo": "COG",
    "Democratic Republic of the Congo": "COD",
    "Russia": "RUS",
    "Syria": "SYR",
    "Myanmar": "MMR",
    "Laos": "LAO",
    "Cape Verde": "CPV",
    "Cabo Verde": "CPV",
    "Eswatini": "SWZ",
    "Swaziland": "SWZ",
    "Czech Republic": "CZE",
    "Kosovo": "XKX",
    "Palestine": "PSE",
    "Hong Kong": "HKG",
    "Macau": "MAC",
    "Taiwan": "TWN",
    "Vatican City": "VAT",
    "Bahamas": "BHS",
    "Gambia": "GMB",
    "Bolivia": "BOL",
    "Venezuela": "VEN",
}


def normalize_country_to_iso3(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    key = name.strip()
    if key in ISO_ALIASES:
        return ISO_ALIASES[key]
    try:
        c = pycountry.countries.lookup(key)
        return c.alpha_3
    except Exception:
        return ""


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def impute_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        if "region" in df.columns:
            regional = df.groupby("region")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(regional)
        df[col] = df[col].fillna(df[col].median())
    return df


def clean_borders_column(df: pd.DataFrame) -> pd.DataFrame:
    if "borders" in df.columns:
        df["borders"] = df["borders"].astype(str).replace({"nan": ""})
    return df


def add_iso_codes(df: pd.DataFrame) -> pd.DataFrame:
    df["iso3"] = df["name"].map(normalize_country_to_iso3)
    return df


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def validate(df: pd.DataFrame) -> None:
    if "name" in df.columns and df["name"].isna().any():
        missing = int(df["name"].isna().sum())
        print(f"Warning: {missing} rows missing country name")
    if "GDP Growth" in df.columns:
        out = df["GDP Growth"].dropna()
        if not out.empty and ((out < -50) | (out > 200)).any():
            print("Warning: extreme GDP Growth values detected")


def main(raw_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(raw_path)

    expected = [
        "name","currency","capital","languages","latitude","longitude","area",
        "region","subregion","borders",
    ] + NUMERIC_COLS
    present = [c for c in expected if c in df.columns]
    df = df[present].copy()

    df = trim_whitespace(df)
    df = coerce_numeric(df, NUMERIC_COLS + ["latitude", "longitude", "area"])
    df = clean_borders_column(df)
    df = impute_numeric(df, NUMERIC_COLS)
    df = add_iso_codes(df)

    if "name" in df.columns:
        all_nan_metrics = df[NUMERIC_COLS].isna().all(axis=1)
        df = df.loc[~(df["name"].isna() & all_nan_metrics)].reset_index(drop=True)

    validate(df)

    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception as e:
        print(f"Parquet write failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=RAW_PATH)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    main(args.raw, args.out)