import matplotlib.pyplot as plt


def plot_correlation_matrix(correlation):

    plt.figure(figsize=(8, 6))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "Air Quality Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.show()


def plot_explained_variance(
    explained_variance
):

    components = [
        f"PC{i + 1}"
        for i in range(
            len(explained_variance)
        )
    ]

    variance_percent = (
        explained_variance * 100
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        components,
        variance_percent
    )

    plt.title(
        "Explained Variance by Principal Component"
    )

    plt.xlabel(
        "Principal Components"
    )

    plt.ylabel(
        "Explained Variance (%)"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.show()


def plot_cumulative_variance(
    cumulative_variance,
    threshold=0.90
):

    components = range(
        1,
        len(cumulative_variance) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        components,
        cumulative_variance * 100,
        marker="o"
    )

    plt.axhline(
        threshold * 100,
        linestyle="--"
    )

    plt.title(
        "Cumulative Explained Variance"
    )

    plt.xlabel(
        "Number of Principal Components"
    )

    plt.ylabel(
        "Cumulative Variance (%)"
    )

    plt.grid(alpha=0.3)

    plt.show()


def plot_pca_2d(reduced_df):

    if (
        "PC1" not in reduced_df.columns
        or "PC2" not in reduced_df.columns
    ):

        print(
            "\nPC1 and PC2 are not both available."
        )

        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        reduced_df["PC1"],
        reduced_df["PC2"],
        s=20,
        alpha=0.6
    )

    plt.title(
        "Air Quality Data in PCA Space"
    )

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.grid(alpha=0.3)

    plt.show()


def plot_loadings(loadings):

    if "PC1" not in loadings.columns:
        return

    pc1 = loadings["PC1"]

    plt.figure(figsize=(8, 5))

    pc1.plot(
        kind="bar"
    )

    plt.title(
        "Feature Contributions to PC1"
    )

    plt.xlabel(
        "Air Quality Features"
    )

    plt.ylabel(
        "Loading"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()