from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error in percent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0
    return float(
        np.mean(
            np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
        )
        * 100
    )


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate RMSE, R2 and MAPE."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
    }


def build_model_searches(preprocessor, cv: KFold) -> Dict[str, GridSearchCV]:
    """Create CV searches for Linear, Ridge and Lasso regression."""
    models = {
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

    for name, (estimator, params) in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
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


def fit_and_evaluate(
    searches: Dict[str, GridSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV]]:
    """Fit models, select hyperparameters using CV and evaluate on test data."""
    rows = []

    for name, search in searches.items():
        search.fit(X_train, y_train)
        pred = search.predict(X_test)

        metrics = regression_metrics(y_test, pred)

        rows.append(
            {
                "Model": name,
                **metrics,
                "Best_CV_RMSE": float(-search.best_score_),
                "Best_Params": str(search.best_params_),
            }
        )

    return pd.DataFrame(rows), searches
