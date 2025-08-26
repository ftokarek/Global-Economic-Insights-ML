import os
from typing import Optional

import pandas as pd
import streamlit as st

PROCESSED_CSV = os.environ.get(PROCESSED_CSV, data/processed/world_economics_clean.csv)
RAW_CSV = os.environ.get(RAW_CSV, data/raw/world_economics.csv)

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV)
        df[source] = processed
        return df
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        df[source] = raw
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def list_countries(df: pd.DataFrame) -> list[str]:
    if df.empty or name not in df.columns:
        return []
    return sorted([c for c in df[name].dropna().unique().tolist()])
