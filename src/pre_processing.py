import pandas as pd
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from utils import get_labels_by_indexes


def comput_PCA(features: pd.DataFrame) -> dict[str, float]:
    """Get the features sorted by variance ratio with PCA algorithm

    Args:
        features (pd.DataFrame): standardized training data

    Returns:
        dict[str, float]: key - the feature name; value - the variance ratio
    """
    features = features.drop("label", axis=1)

    pca: PCA = PCA(n_components=features.shape[1])

    pca.fit_transform(features)

    return dict(zip(pca.feature_names_in_, pca.explained_variance_ratio_))  # type: ignore


def comput_kruskal(
    features: pd.DataFrame,
) -> dict[str, float]:
    """Comput kruskal wallis H value for each feature

    Args:
        features (pd.DataFrame): standardized training data
        labels_indexes (dict[str, pd.Index]): dataframe labels indexes

    Returns:
        dict[str, float]: return the sorted H value by feature
    """

    labels_indexes: dict[str, pd.Index] = get_labels_by_indexes(features)

    kruskal_wallis_features = {}

    features = features.drop("label", axis=1)

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
