import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_and_standardize_data(target_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read data from data file, split in train and test groups and standardize the data with z-score

    Returns:
        tuple[pd.DataFrame, dict[(str) Class, (pd.Index) Dataframe Indexes]]:
            Return the standardized training data and the indexes of the samples grouped by class
    """

    # read the data
    data: pd.DataFrame = pd.read_csv("../data/dados.csv")

    # split the data, 80% training, 20% test
    data_train, data_test = train_test_split(
        data, test_size=0.2, stratify=data.label, random_state=42
    )

    def standardize(data):
        data_df: pd.DataFrame = pd.DataFrame(data)

        # get the only the features (remove song name and label)
        features: pd.DataFrame = data_df.iloc[:, 1:-1]

        # standardize the data with z-score
        standarized_features: pd.DataFrame = pd.DataFrame(
            StandardScaler().fit_transform(features),
            columns=features.columns,
            index=features.index,
        )

        data_df.loc[data_df.label != target_class, "label"] = "other"

        return pd.concat([standarized_features, data_df.label], axis=1)

    return standardize(data_train), standardize(data_test)
