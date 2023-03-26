import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def comput_PCA(features: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """Get the features sorted by variance ratio with PCA algorithm

    Args:
        features (pd.DataFrame): standardized training data
        n_componentes (int): number of components (features) to kept

    Returns:
        DataFrame: transformed data
    """

    pca: PCA = PCA(n_components=n_components, svd_solver="full")

    transformed_data: np.ndarray = pca.fit_transform(features)

    # print(pca.explained_variance_ratio_)

    return pd.DataFrame(
        transformed_data, columns=pca.get_feature_names_out(), index=features.index
    )


def comput_LDA(
    features: pd.DataFrame, labels: pd.Series, n_components: int
) -> pd.DataFrame:
    lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis(
        n_components=n_components
    )

    transformed_data: np.ndarray = lda.fit_transform(X=features, y=labels)

    return pd.DataFrame(
        transformed_data, columns=lda.get_feature_names_out(), index=features.index
    )


def comput_kruskal(
    features: pd.DataFrame, n_components: int, labels_indexes: dict[str, pd.Index]
) -> pd.DataFrame:
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

    return features[list(kruskal_wallis_features.keys())[:n_components]]
