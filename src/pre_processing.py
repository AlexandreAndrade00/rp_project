import numpy as np
from kruskal_wallis import KruskalWallis
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
    data_X: np.ndarray, data_y: np.ndarray | None, model: LinearDiscriminantAnalysis | None
) -> tuple[LinearDiscriminantAnalysis, np.ndarray]:
    
    if model is None:
        if  data_y is None:
            raise ValueError("If no model is given, it is necessary the labels to train one") 

        lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis()

        lda.fit(X=data_X, y=data_y)
    else:
        lda = model

    return lda, lda.transform(data_X)


def comput_kruskal(X: np.ndarray, y: np.ndarray | None, model: KruskalWallis | None, n_components: int = 50) -> tuple[KruskalWallis, np.ndarray]:
    """Comput kruskal wallis H value for each feature

    Args:
        features (pd.DataFrame): standardized training data
        labels_indexes (dict[str, pd.Index]): dataframe labels indexes

    Returns:
        dict[str, float]: return the sorted H value by feature
    """
    if model is None:
        if  y is None:
            raise ValueError("If no model is given, it is necessary the labels to train one") 

        kw: KruskalWallis = KruskalWallis(n_components)

        kw.fit(X=X, y=y)
    else:
        kw = model

    return kw, kw.transform(X)