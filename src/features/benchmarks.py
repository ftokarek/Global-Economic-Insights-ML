from __future__ import annotations
import pandas as pd

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


def compute_global_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[f"{col}__pctile"] = out[col].rank(pct=True) * 100.0
    return out


def compute_region_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if "region" not in df.columns:
        return pd.DataFrame()
    agg = (
        df.groupby("region")[NUMERIC_COLS]
        .median()
        .rename(columns=lambda c: f"{c}__region_median")
        .reset_index()
    )
    return agg


def attach_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    base = compute_global_percentiles(df)
    if "region" in df.columns:
        agg = compute_region_aggregates(base)
        base = base.merge(agg, on="region", how="left")
    return base