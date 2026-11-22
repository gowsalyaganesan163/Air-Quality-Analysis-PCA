def dataset_overview(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("Number of Rows:", df.shape[0])

    print("Number of Columns:", df.shape[1])

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())


def numerical_statistics(features):

    print("\n========== NUMERICAL STATISTICS ==========")

    print(features.describe())


def correlation_analysis(features):

    print("\n========== CORRELATION MATRIX ==========")

    correlation = features.corr()

    print(correlation.round(2))

    return correlation