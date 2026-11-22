import numpy as np


def analyze_components(loadings):

    print(
        "\n========== COMPONENT ANALYSIS =========="
    )

    for component in loadings.columns:

        print(f"\n{component}")

        component_values = loadings[
            component
        ].sort_values(
            key=abs,
            ascending=False
        )

        for feature, value in component_values.items():

            print(
                f"{feature}: {value:.3f}"
            )

        strongest_feature = (
            component_values.index[0]
        )

        strongest_value = (
            component_values.iloc[0]
        )

        print(
            f"Strongest contribution: "
            f"{strongest_feature} "
            f"({strongest_value:.3f})"
        )


def summarize_variance(
    explained_variance,
    cumulative_variance
):

    print(
        "\n========== VARIANCE SUMMARY =========="
    )

    for i in range(len(explained_variance)):

        print(
            f"PC{i + 1} retains "
            f"{explained_variance[i] * 100:.2f}% "
            f"of the variance."
        )

    total = cumulative_variance[-1] * 100

    print(
        f"\nTotal variance represented: "
        f"{total:.2f}%"
    )


def save_reduced_data(
    reduced_df,
    file_name="air_quality_pca_results.csv"
):

    reduced_df.to_csv(
        file_name,
        index=False
    )

    print(
        f"\nReduced PCA data saved to: "
        f"{file_name}"
    )