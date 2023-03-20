import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kruskal


def main():
    standarized_features, labels_indexes = read_and_standardize_data()

    print(comput_PCA(standarized_features))

    print(comput_kruskal(standarized_features, labels_indexes))


def read_and_standardize_data() -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    """Read data from data file, split in train and test groups and standardize the data with z-score

    Returns:
        tuple[pd.DataFrame, dict[(str) Class, (pd.Index) Dataframe Indexes]]:
            Return the standardized training data and the indexes of the samples grouped by class
    """

    # read the data
    data: pd.DataFrame = pd.read_csv("data/dados.csv")

    # split the data, 80% training, 20% test
    data_train, data_test = train_test_split(data, test_size=0.20, stratify=data.label)

    data_train = pd.DataFrame(data_train)

    # get the only the features (remove song name and label)
    features: pd.DataFrame = data_train.iloc[:, 1:-1]

    # get dataframe indexes splitted by label
    labels_indexes: dict[str, pd.Index] = get_labels_by_indexes(data_train)

    # standardize the data with z-score
    standarized_features: pd.DataFrame = pd.DataFrame(
        StandardScaler().fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    return standarized_features, labels_indexes


def comput_PCA(features: pd.DataFrame) -> dict[str, float]:
    """Get the features sorted by variance ratio with PCA algorithm

    Args:
        features (pd.DataFrame): standardized training data

    Returns:
        dict[str, float]: key - the feature name; value - the variance ratio
    """
    pca: PCA = PCA(n_components=features.shape[1])

    pca.fit_transform(features)

    return dict(zip(pca.feature_names_in_, pca.explained_variance_ratio_))  # type: ignore


def comput_kruskal(
    features: pd.DataFrame, labels_indexes: dict[str, pd.Index]
) -> dict[str, float]:
    """Comput kruskal wallis H value for each feature

    Args:
        features (pd.DataFrame): standardized training data
        labels_indexes (dict[str, pd.Index]): dataframe labels indexes

    Returns:
        dict[str, float]: return the sorted H value by feature
    """

    kruskal_wallis_features = {}

    # apply kruskal wallis for each feature
    for feature in features:
        classes_values = ()

        # split the samples by labels/classes
        for indexes in labels_indexes.values():
            classes_values = classes_values + (features[feature].loc[indexes].values,)

        try:
            # run kruskal wallis
            result = kruskal(*classes_values)

            # 99% confidence interval - p-value < 0.01
            if result[1] < 0.01:
                kruskal_wallis_features[feature] = result[0]

        # equal values case
        except ValueError:
            kruskal_wallis_features[feature] = 0

    # sort features by H value
    kruskal_wallis_features = dict(
        sorted(kruskal_wallis_features.items(), key=lambda x: x[1], reverse=True)
    )

    return kruskal_wallis_features


def get_labels_by_indexes(data: pd.DataFrame) -> dict[str, pd.Index]:
    """Get the dataframe indexes splitted by labels

    Args:
        data (pd.DataFrame): data

    Returns:
        dict[str, pd.Index]: str - label; value - dataframe indexes
    """
    labels_indexes: dict[str, pd.Index] = {}

    for label in data.label.unique():
        labels_indexes[label] = data[data.label == label].index

    return labels_indexes


if __name__ == "__main__":
    main()
