import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standardize_data(features):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(features)

    print("\n========== STANDARDIZATION ==========")

    print("StandardScaler applied.")

    return scaled_data, scaler


def apply_pca(scaled_data):

    # Apply PCA to all available components
    pca = PCA()

    principal_components = pca.fit_transform(
        scaled_data
    )

    print("\n========== PCA APPLIED ==========")

    print(
        "Original number of features:",
        scaled_data.shape[1]
    )

    print(
        "Number of principal components:",
        principal_components.shape[1]
    )

    return pca, principal_components


def calculate_explained_variance(pca):

    explained_variance = (
        pca.explained_variance_ratio_
    )

    cumulative_variance = (
        explained_variance.cumsum()
    )

    print("\n========== EXPLAINED VARIANCE ==========")

    for i in range(len(explained_variance)):

        print(
            f"PC{i + 1}: "
            f"{explained_variance[i] * 100:.2f}%"
            f" | Cumulative: "
            f"{cumulative_variance[i] * 100:.2f}%"
        )

    return explained_variance, cumulative_variance


def choose_components(
    cumulative_variance,
    threshold=0.90
):

    number_of_components = (
        cumulative_variance >= threshold
    ).argmax() + 1

    print(
        f"\nComponents needed to retain "
        f"{threshold * 100:.0f}% variance:",
        number_of_components
    )

    return number_of_components


def create_reduced_data(
    scaled_data,
    number_of_components
):

    pca_reduced = PCA(
        n_components=number_of_components
    )

    reduced_data = pca_reduced.fit_transform(
        scaled_data
    )

    component_names = [
        f"PC{i + 1}"
        for i in range(number_of_components)
    ]

    reduced_df = pd.DataFrame(
        reduced_data,
        columns=component_names
    )

    print(
        "\n========== REDUCED DATA =========="
    )

    print(
        "Original dimensions:",
        scaled_data.shape[1]
    )

    print(
        "Reduced dimensions:",
        number_of_components
    )

    print("\nFirst 5 rows:")

    print(reduced_df.head())

    return reduced_df, pca_reduced


def calculate_loadings(
    pca,
    feature_names
):

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[
            f"PC{i + 1}"
            for i in range(pca.n_components_)
        ]
    )

    print(
        "\n========== PCA LOADINGS =========="
    )

    print(
        loadings.round(3)
    )

    return loadings