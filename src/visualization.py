from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_target_and_feature_distributions(
    df: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    for col in numeric_cols:
        plt.figure(figsize=(7, 5))
        plt.hist(df[col].dropna(), bins=25)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"distribution_{col}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    groups = [
        df.loc[df["species"] == species, "body_mass_g"].dropna().values
        for species in df["species"].dropna().unique()
    ]
    labels = list(df["species"].dropna().unique())
    plt.boxplot(groups, labels=labels)
    plt.xlabel("Species")
    plt.ylabel("body_mass_g")
    plt.title("Body mass by penguin species")
    plt.tight_layout()
    plt.savefig(output_dir / "body_mass_by_species.png", dpi=150)
    plt.close()


def save_scatter_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
    ]

    for col in numeric_features:
        plt.figure(figsize=(7, 5))
        for species in df["species"].dropna().unique():
            subset = df[df["species"] == species]
            plt.scatter(
                subset[col],
                subset["body_mass_g"],
                label=species,
                alpha=0.7,
            )
        plt.xlabel(col)
        plt.ylabel("body_mass_g")
        plt.title(f"Body mass vs {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_body_mass_vs_{col}.png", dpi=150)
        plt.close()


def save_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    corr = df[cols].corr()
    corr.to_csv(output_dir.parent / "correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()

    return corr


def save_vif_plot(vif_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.barh(vif_df["Feature"], vif_df["VIF"])
    plt.xlabel("VIF")
    plt.ylabel("Feature")
    plt.title("Variance Inflation Factor (VIF)")
    plt.tight_layout()
    plt.savefig(output_dir / "vif.png", dpi=150)
    plt.close()


def save_pca_variance_plot(
    explained_variance_ratio,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = explained_variance_ratio
    cumulative = ratios.cumsum()
    components = list(range(1, len(ratios) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(components, ratios, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree plot")
    plt.xticks(components)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scree_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(components, cumulative, marker="o")
    plt.axhline(0.95, linestyle="--", label="95% variance")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance")
    plt.xticks(components)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_explained_variance.png", dpi=150)
    plt.close()


def save_model_comparison(
    before_pca: pd.DataFrame,
    after_pca: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(before_pca["Model"])
    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        [i - width / 2 for i in x],
        before_pca["RMSE"],
        width=width,
        label="Before PCA",
    )
    plt.bar(
        [i + width / 2 for i in x],
        after_pca["RMSE"],
        width=width,
        label="After PCA",
    )
    plt.xticks(x, models)
    plt.ylabel("RMSE")
    plt.title("RMSE comparison before and after PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_before_after_pca.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(
        [i - width / 2 for i in x],
        before_pca["R2"],
        width=width,
        label="Before PCA",
    )
    plt.bar(
        [i + width / 2 for i in x],
        after_pca["R2"],
        width=width,
        label="After PCA",
    )
    plt.xticks(x, models)
    plt.ylabel("R²")
    plt.title("R² comparison before and after PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "r2_before_after_pca.png", dpi=150)
    plt.close()
