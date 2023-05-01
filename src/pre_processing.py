import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def comput_PCA(
    features: np.ndarray, model: PCA | None = None
) -> tuple[PCA, np.ndarray]:
    """Get the features sorted by variance ratio with PCA algorithm

    Args:
        features (pd.DataFrame): standardized training data
        n_componentes (int): number of components (features) to kept

    Returns:
        DataFrame: transformed data
    """
    if model is None:
        pca: PCA = PCA(n_components="mle", svd_solver="full")
        pca.fit(features)
    else:
        pca = model

    # print(pca.explained_variance_ratio_)

    return pca, pca.transform(features)


def comput_LDA(
    data_X: np.ndarray, data_y: np.ndarray, model: LinearDiscriminantAnalysis | None
) -> tuple[LinearDiscriminantAnalysis, np.ndarray]:
    if model is None:
        lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis()

        lda.fit(X=data_X, y=data_y)
    else:
        lda = model

    return lda, lda.transform(data_X)


def comput_kruskal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Comput kruskal wallis H value for each feature

    Args:
        features (pd.DataFrame): standardized training data
        labels_indexes (dict[str, pd.Index]): dataframe labels indexes

    Returns:
        dict[str, float]: return the sorted H value by feature
    """

    kruskal_wallis_features = {}

    labels = np.unique(y)

    # apply kruskal wallis for each feature
    for i in range(X.shape[1]):
        classes_values = ()

        this_feature = X[:, i]

        # split the samples by labels/classes
        for label in labels:
            classes_values = classes_values + (this_feature[label == y],)

        try:
            # run kruskal wallis
            result = kruskal(*classes_values)

            # 99% confidence interval - p-value < 0.01
            if result[1] < 0.01:
                kruskal_wallis_features[i] = result[0]
            else:
                kruskal_wallis_features[i] = 0

        # equal values case
        except ValueError:
            kruskal_wallis_features[i] = 0

    # sort features by H value
    kruskal_wallis_features = dict(
        sorted(kruskal_wallis_features.items(), key=lambda x: x[1], reverse=True)
    )

    return X[list(kruskal_wallis_features.keys())[:n_components]]
