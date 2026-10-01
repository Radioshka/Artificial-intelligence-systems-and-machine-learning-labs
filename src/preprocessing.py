from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "body_mass_g"

NUMERIC_FEATURES = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
]

CATEGORICAL_FEATURES = [
    "species",
    "island",
    "sex",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the penguin CSV and validate the expected columns."""
    path = Path(path)
    df = pd.read_csv(path)

    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and regression target."""
    # A regression model cannot be trained with a missing target.
    # We remove only rows where body_mass_g is missing.
    clean_df = df.dropna(subset=[TARGET]).copy()
    X = clean_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = clean_df[TARGET].copy()
    return X, y


def build_model_preprocessor() -> ColumnTransformer:
    """
    Preprocessor for Linear/Ridge/Lasso models.

    Numeric variables:
      median imputation + standardization.

    Categorical variables:
      most-frequent imputation + one-hot encoding.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_pca_preprocessor() -> Pipeline:
    """
    Preprocessor for PCA.

    First, missing values are handled and categorical variables are encoded.
    Then all resulting variables are standardized before PCA.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return Pipeline(
        steps=[
            ("columns", column_transformer),
            ("scaler", StandardScaler()),
        ]
    )


def impute_numeric_for_vif(X_train: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric predictors using training-set medians for VIF."""
    X_num = X_train[NUMERIC_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(X_num)
    return pd.DataFrame(values, columns=NUMERIC_FEATURES, index=X_num.index)
