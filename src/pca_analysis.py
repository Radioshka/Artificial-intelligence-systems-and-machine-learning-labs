from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from .models import mape, regression_metrics


def make_pca_model_searches(
    pca_preprocessor,
    cv: KFold,
) -> Dict[str, GridSearchCV]:
    """
    Build regression pipelines where PCA is applied after standardization.

    PCA retains 95% of the variance. This value is selected from the
    training data through PCA itself.
    """
    model_specs = {
        "LinearRegression": (
            LinearRegression(),
            {},
        ),
        "Ridge": (
            Ridge(),
            {"model__alpha": np.logspace(-3, 3, 13)},
        ),
        "Lasso": (
            Lasso(max_iter=100000),
            {"model__alpha": np.logspace(-3, 1, 12)},
        ),
    }

    searches = {}

    for name, (model, params) in model_specs.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", pca_preprocessor),
                ("pca", PCA(n_components=0.95)),
                ("model", model),
            ]
        )

        searches[name] = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )

    return searches


def fit_and_evaluate_pca(
    searches: Dict[str, GridSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV]]:
    rows = []

    for name, search in searches.items():
        search.fit(X_train, y_train)
        pred = search.predict(X_test)

        pca = search.best_estimator_.named_steps["pca"]

        metrics = regression_metrics(y_test, pred)

        rows.append(
            {
                "Model": name,
                **metrics,
                "Best_CV_RMSE": float(-search.best_score_),
                "PCA_components": int(pca.n_components_),
                "PCA_explained_variance": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "Best_Params": str(search.best_params_),
            }
        )

    return pd.DataFrame(rows), searches
