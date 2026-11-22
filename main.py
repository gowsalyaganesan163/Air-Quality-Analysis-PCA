from tkinter import Tk, filedialog

from data_loader import (
    load_data,
    clean_data,
    select_air_quality_features
)

from eda import (
    dataset_overview,
    numerical_statistics,
    correlation_analysis
)

from pca_model import (
    standardize_data,
    apply_pca,
    calculate_explained_variance,
    choose_components,
    create_reduced_data,
    calculate_loadings
)

from analysis import (
    analyze_components,
    summarize_variance,
    save_reduced_data
)

from visualization import (
    plot_correlation_matrix,
    plot_explained_variance,
    plot_cumulative_variance,
    plot_pca_2d,
    plot_loadings
)


def select_file():

    root = Tk()

    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Air Quality CSV File",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return file_path


def main():

    print("==========================================")
    print("       AIR QUALITY ANALYSIS USING PCA")
    print("==========================================")

    # -----------------------------------
    # 1. Select file
    # -----------------------------------

    file_path = select_file()

    if not file_path:

        print("\nNo file selected.")

        return

    # -----------------------------------
    # 2. Load data
    # -----------------------------------

    df = load_data(file_path)

    # -----------------------------------
    # 3. Clean data
    # -----------------------------------

    df = clean_data(df)

    # -----------------------------------
    # 4. Dataset overview
    # -----------------------------------

    dataset_overview(df)

    # -----------------------------------
    # 5. Select air-quality features
    # -----------------------------------

    features = select_air_quality_features(
        df
    )

    # -----------------------------------
    # 6. Statistics
    # -----------------------------------

    numerical_statistics(
        features
    )

    # -----------------------------------
    # 7. Correlation
    # -----------------------------------

    correlation = correlation_analysis(
        features
    )

    # -----------------------------------
    # 8. Correlation visualization
    # -----------------------------------

    plot_correlation_matrix(
        correlation
    )

    # -----------------------------------
    # 9. Standardization
    # -----------------------------------

    scaled_data, scaler = standardize_data(
        features
    )

    # -----------------------------------
    # 10. Apply PCA
    # -----------------------------------

    pca, principal_components = apply_pca(
        scaled_data
    )

    # -----------------------------------
    # 11. Explained variance
    # -----------------------------------

    (
        explained_variance,
        cumulative_variance
    ) = calculate_explained_variance(
        pca
    )

    # -----------------------------------
    # 12. Variance graphs
    # -----------------------------------

    plot_explained_variance(
        explained_variance
    )

    plot_cumulative_variance(
        cumulative_variance
    )

    # -----------------------------------
    # 13. Choose components
    # -----------------------------------

    number_of_components = choose_components(
        cumulative_variance,
        threshold=0.90
    )

    # -----------------------------------
    # 14. Reduce dimensions
    # -----------------------------------

    (
        reduced_df,
        reduced_pca
    ) = create_reduced_data(
        scaled_data,
        number_of_components
    )

    # -----------------------------------
    # 15. Calculate loadings
    # -----------------------------------

    loadings = calculate_loadings(
        reduced_pca,
        features.columns
    )

    # -----------------------------------
    # 16. Analyze components
    # -----------------------------------

    analyze_components(
        loadings
    )

    summarize_variance(
        explained_variance,
        cumulative_variance
    )

    # -----------------------------------
    # 17. PCA 2D visualization
    # -----------------------------------

    plot_pca_2d(
        reduced_df
    )

    # -----------------------------------
    # 18. Loading visualization
    # -----------------------------------

    plot_loadings(
        loadings
    )

    # -----------------------------------
    # 19. Save results
    # -----------------------------------

    save_reduced_data(
        reduced_df
    )

    print("\n==========================================")
    print("           PCA ANALYSIS COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()




