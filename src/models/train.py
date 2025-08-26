from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# LightGBM is optional; we import inside function to avoid hard dependency at import time.

TARGET = "GDP Growth"
NUMERIC_COLS = [
    "GDP",
    "Interest Rate",
    "Inflation Rate",
    "Jobless Rate",
    "Gov. Budget",
    "Debt/GDP",
    "Current Account",
    "Population",
]


@dataclass
class ModelResult:
    name: str
    fold_metrics: List[Dict[str, float]]
    cv_metrics: Dict[str, float]
    feature_names: List[str]
    importances: List[float] | None = None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def _numeric_preprocessor(numeric: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        ]
    )


def train_elasticnet(df: pd.DataFrame, n_splits: int = 5) -> ModelResult:
    data = df.dropna(subset=[TARGET])
    X: pd.DataFrame = data[NUMERIC_COLS].copy()
    y = data[TARGET].values

    pre = _numeric_preprocessor(NUMERIC_COLS)
    model = ElasticNet(alpha=0.1, l1_ratio=0.3, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    kf = KFold(n_splits=min(max(2, n_splits), len(data)), shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []
    for train_idx, test_idx in kf.split(X):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        fold_metrics.append(_compute_metrics(y[test_idx], pred))

    cv = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()}
    return ModelResult(name="ElasticNet", fold_metrics=fold_metrics, cv_metrics=cv, feature_names=NUMERIC_COLS)


def train_lightgbm(df: pd.DataFrame, n_splits: int = 5) -> ModelResult:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        raise RuntimeError(f"LightGBM unavailable: {e}")

    data = df.dropna(subset=[TARGET])
    X: pd.DataFrame = data[NUMERIC_COLS].copy()
    y = data[TARGET].values

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
        ]
    )
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=15,
        min_data_in_leaf=8,
        min_gain_to_split=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    kf = KFold(n_splits=min(max(2, n_splits), len(data)), shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []
    importances_accum = np.zeros(len(NUMERIC_COLS), dtype=float)

    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        pipe.fit(X_train, y[train_idx])
        pred = pipe.predict(X_test)
        fold_metrics.append(_compute_metrics(y[test_idx], pred))
        gbm = pipe.named_steps["model"]
        if hasattr(gbm, "feature_importances_"):
            importances_accum += gbm.feature_importances_

    importances = (importances_accum / max(1, len(fold_metrics))).tolist()
    cv = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()}
    return ModelResult(name="LightGBM", fold_metrics=fold_metrics, cv_metrics=cv, feature_names=NUMERIC_COLS, importances=importances)


def fit_elasticnet_full(df: pd.DataFrame):
    data = df.dropna(subset=[TARGET]).copy()
    X: pd.DataFrame = data[NUMERIC_COLS].copy()
    y = data[TARGET].values
    pre = _numeric_preprocessor(NUMERIC_COLS)
    model = ElasticNet(alpha=0.1, l1_ratio=0.3, random_state=42)
    pipe = Pipeline(steps=[('pre', pre), ('model', model)])
    pipe.fit(X, y)
    return pipe


def fit_lightgbm_full(df: pd.DataFrame):
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        raise RuntimeError(f'LightGBM unavailable: {e}')
    data = df.dropna(subset=[TARGET]).copy()
    X: pd.DataFrame = data[NUMERIC_COLS].copy()
    y = data[TARGET].values
    pre = ColumnTransformer(transformers=[('num', SimpleImputer(strategy='median'), NUMERIC_COLS)])
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=15,
        min_data_in_leaf=8,
        min_gain_to_split=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline(steps=[('pre', pre), ('model', model)])
    pipe.fit(X, y)
    return pipe