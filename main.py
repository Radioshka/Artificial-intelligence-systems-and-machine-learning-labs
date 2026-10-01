from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.models import build_model_searches, fit_and_evaluate
from src.pca_analysis import make_pca_model_searches, fit_and_evaluate_pca
from src.preprocessing import (
    TARGET,
    build_model_preprocessor,
    build_pca_preprocessor,
    impute_numeric_for_vif,
    load_data,
    split_features_target,
)
from src.visualization import (
    save_correlation_matrix,
    save_model_comparison,
    save_pca_variance_plot,
    save_scatter_plots,
    save_target_and_feature_distributions,
    save_vif_plot,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "penguins.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def calculate_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for numeric predictors."""
    rows = []

    for i, feature in enumerate(X_numeric.columns):
        rows.append(
            {
                "Feature": feature,
                "VIF": variance_inflation_factor(
                    X_numeric.values, i
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PENGUIN REGRESSION + PCA LAB")
    print("=" * 70)

    # 1. Load data
    df = load_data(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nTarget:", TARGET)
    print("\nTarget statistics:")
    print(df[TARGET].describe())

    # Initial visual analysis
    save_target_and_feature_distributions(df, FIGURES_DIR)
    save_scatter_plots(df, FIGURES_DIR)
    corr = save_correlation_matrix(df, FIGURES_DIR)

    print("\nCorrelation matrix:")
    print(corr.round(3))

    # 2. Split data
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    # 3. VIF on numeric predictors using only training data
    X_train_num = impute_numeric_for_vif(X_train)
    vif_df = calculate_vif(X_train_num)
    vif_df.to_csv(RESULTS_DIR / "vif.csv", index=False)
    save_vif_plot(vif_df, FIGURES_DIR)

    print("\nVIF:")
    print(vif_df.round(3))

    # 4. Models before PCA
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    model_preprocessor = build_model_preprocessor()
    searches_before = build_model_searches(model_preprocessor, cv)

    before_results, searches_before = fit_and_evaluate(
        searches_before,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    before_results.to_csv(
        RESULTS_DIR / "results_before_pca.csv",
        index=False,
    )

    print("\nResults BEFORE PCA:")
    print(
        before_results[
            ["Model", "RMSE", "R2", "MAPE_%", "Best_CV_RMSE"]
        ].round(4)
    )

    # 5. PCA + models
    pca_preprocessor = build_pca_preprocessor()
    searches_after = make_pca_model_searches(pca_preprocessor, cv)

    after_results, searches_after = fit_and_evaluate_pca(
        searches_after,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    after_results.to_csv(
        RESULTS_DIR / "results_after_pca.csv",
        index=False,
    )

    print("\nResults AFTER PCA:")
    print(
        after_results[
            [
                "Model",
                "RMSE",
                "R2",
                "MAPE_%",
                "Best_CV_RMSE",
                "PCA_components",
                "PCA_explained_variance",
            ]
        ].round(4)
    )

    # 6. PCA plots using the best PCA model's fitted transformation
    reference_model = searches_after["LinearRegression"].best_estimator_
    fitted_pca = reference_model.named_steps["pca"]

    save_pca_variance_plot(
        fitted_pca.explained_variance_ratio_,
        FIGURES_DIR,
    )

    # 7. Combined comparison
    combined = pd.concat(
        [
            before_results.assign(Stage="Before PCA"),
            after_results.assign(Stage="After PCA"),
        ],
        ignore_index=True,
    )

    combined.to_csv(
        RESULTS_DIR / "results.csv",
        index=False,
    )

    save_model_comparison(
        before_results,
        after_results,
        FIGURES_DIR,
    )

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved figures to:", FIGURES_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
