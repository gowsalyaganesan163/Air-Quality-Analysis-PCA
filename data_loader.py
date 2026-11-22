import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)

    print("\n========== DATA LOADED ==========")
    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    return df


def clean_data(df):
    print("\n========== DATA CLEANING ==========")

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    print("Shape after removing empty rows/columns:", df.shape)

    return df


def select_air_quality_features(df):
    """
    Select common air-quality variables for PCA.
    AQI is excluded because it is usually a derived measure.
    """

    possible_features = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ]

    available_features = [
        column
        for column in possible_features
        if column in df.columns
    ]

    # Handle lowercase/alternative names if needed
    if len(available_features) < 2:

        column_map = {
            column.lower().replace(" ", ""): column
            for column in df.columns
        }

        alternative_names = {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO2",
            "so2": "SO2",
            "co": "CO",
            "o3": "O3"
        }

        available_features = []

        for key, standard_name in alternative_names.items():

            if key in column_map:
                available_features.append(
                    column_map[key]
                )

    if len(available_features) < 2:

        raise ValueError(
            "\nCould not find enough air-quality features.\n"
            "Expected columns such as: PM2.5, PM10, NO2, SO2, CO, O3."
        )

    features = df[available_features].copy()

    # Convert values to numeric
    for column in features.columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    # Replace infinity
    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # Fill missing values using median
    for column in features.columns:

        features[column] = features[column].fillna(
            features[column].median()
        )

    print("\n========== PCA FEATURES ==========")

    print("Features selected:")

    for column in features.columns:
        print("-", column)

    print(
        "\nNumber of features:",
        len(features.columns)
    )

    return features