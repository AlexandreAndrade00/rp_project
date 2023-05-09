import numpy as np
from scipy.stats import kruskal

class KruskalWallis:
    def __init__(self, n_components: int) -> None:
        self.__n_components = n_components

    def fit(self, X:np.ndarray, y:np.ndarray):
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

        self.__selected_features = list(kruskal_wallis_features.keys())[:self.__n_components]

        return self

    def transform(self, X:np.ndarray) -> np.ndarray:
        return X[:, self.__selected_features]