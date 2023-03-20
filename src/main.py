import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kruskal


def main():
    data: pd.DataFrame = pd.read_csv("data/dados.csv")

    data_train, data_test = train_test_split(data, test_size=0.20, stratify=data.label)

    data_train = pd.DataFrame(data_train)

    features: pd.DataFrame = data_train.iloc[:, 1:-1]

    labels_indexes:dict[str, pd.Index] = get_labels_by_indexes(data_train)

    standarized_features: pd.DataFrame = pd.DataFrame(
        StandardScaler().fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    comput_PCA(standarized_features)

    comput_kruskal(standarized_features, labels_indexes)


def comput_PCA(features: pd.DataFrame):
    pca: PCA = PCA(n_components=features.shape[1])

    pca.fit_transform(features)

    print(pca.explained_variance_ratio_)


def comput_kruskal(features: pd.DataFrame, labels_indexes: dict[str, pd.Index]):
    kruskal_wallis_features = {}

    for feature in features:

        classes_values = ()

        for indexes in labels_indexes.values():
            classes_values = classes_values + (features[feature].loc[indexes].values,)

        try:
            kruskal_wallis_features[feature] = kruskal(*classes_values)
        except ValueError:
            kruskal_wallis_features[feature] = (0, 0)

    kruskal_wallis_features = dict(sorted(kruskal_wallis_features.items(), key=lambda x:x[1][0], reverse=True))
    print(kruskal_wallis_features)


def get_labels_by_indexes(data: pd.DataFrame) -> dict[str, pd.Index]:
    labels_indexes:dict[str, pd.Index] = {}

    for label in data.label.unique():
        labels_indexes[label] = data[data.label == label].index

    return labels_indexes


if __name__ == "__main__":
    main()
