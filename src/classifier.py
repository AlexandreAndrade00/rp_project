from typing import Literal
import numpy as np
from kruskal_wallis import KruskalWallis
import pre_processing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class Classifier:
    __feature_reduction_model: PCA | LinearDiscriminantAnalysis | None = None
    __feature_selection_model: KruskalWallis | None = None
    __classifier_model: GaussianNB | OneVsOneClassifier | KNeighborsClassifier | SVC
    __train_X: np.ndarray
    __train_y: np.ndarray
    __target_class: str | None
    __labels: np.ndarray
    __pre_processed_train_X: np.ndarray
    __predicted_labels: np.ndarray

    def __init__(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        self.__train_X = train_X
        self.__train_y = train_y
        self.__pre_processed_train_X = self.__train_X

        self.__labels = np.unique(train_y)

        if self.__labels.size == 2:
            self.__target_class = (
                self.__labels[0] if self.__labels[0] != "other" else self.__labels[1]
            )
        else:
            self.__target_class = None

    def feature_selection(self):
        self.__pre_processed_train_X = self.__feature_selection(
            X=self.__pre_processed_train_X, y=self.__train_y
        )

    def __feature_selection(self, X: np.ndarray, y: np.ndarray | None = None):
        self.__feature_selection_model, result = pre_processing.comput_kruskal(X=X, y=y, model=self.__feature_selection_model)  # type: ignore

        return result

    def feature_reduction(self, method: str) -> None:
        self.__pre_processed_train_X = self.__feature_reduction(
            method, X=self.__pre_processed_train_X, y=self.__train_y
        )

    def __feature_reduction(
        self,
        method: str,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        result: np.ndarray

        match method:
            case "PCA":
                self.__feature_reduction_model, result = pre_processing.comput_PCA(
                    features=X, model=self.__feature_reduction_model  # type: ignore
                )

            case "LDA":
                self.__feature_reduction_model, result = pre_processing.comput_LDA(
                    data_X=X, data_y=y, model=self.__feature_reduction_model  # type: ignore
                )

            case _:
                raise ValueError("Unknown pre process method")

        return result

    def train(self, model: str, **kwargs) -> None:
        match model:
            case "one_vs_all":
                self.__distance_type = kwargs["distance_type"]

                self.__train_one_vs_all_minimum_distance()

            case "gnb":
                self.__train_GNB()

            case "multi_knn":
                self.__train_Knn()

            case "svm":
                self.__train_multi_svm()

            case _:
                raise ValueError(model)

        self.__selected_model: str = model

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        test_data = test_X

        if self.__feature_selection_model is not None:
            test_data = self.__feature_selection(test_X)

        if self.__feature_reduction_model is not None:
            method: str = "PCA" if self.__feature_reduction_model is PCA else "LDA"

            test_data = self.__feature_reduction(
                method,
                test_data,
            )

        match self.__selected_model:
            case "one_vs_all":
                self.__predicted_labels = (
                    self.__predict_one_vs_all_minimum_distance(test_data)
                )

            case "gnb":
                self.__predicted_labels = self.__predict_GNB(test_data)

            case "knn":
                self.__predicted_labels = self.__predict_Knn(test_data)

            case "svm":
                self.__predicted_labels = self.__predict_multi_svm(test_data)

            case _:
                raise NotImplementedError(self.__selected_model)

        return self.__predicted_labels

    # def __train_multi_knn(self, max_iter=300) -> None:
    #     self.n_clusters = self.train_data.label.nunique()
    #     self.max_iter = max_iter
    #     self.centroids = None

    #     data: np.ndarray = (
    #         self.__pre_processed_train_X
    #         if (self.__pre_processed_train_X is not None)
    #         else self.__train_X
    #     )

    #     # Initialize centroids randomly
    #     self.__centroids = data.sample(n=self.n_clusters)

    #     # Iterate until convergence or maximum iterations
    #     for i in range(self.max_iter):
    #         # Assign each point to the nearest centroid
    #         distances = np.sqrt(
    #             ((X - self.centroids.iloc[:, np.newaxis]) ** 2).sum(axis=2)
    #         )
    #         labels = np.argmin(distances, axis=1)

    #         # Update the centroids
    #         new_centroids = []
    #         for j in range(self.n_clusters):
    #             new_centroid = X[labels == j].mean()
    #             new_centroids.append(new_centroid)
    #         new_centroids = pd.concat(new_centroids, axis=1).T
    #         if new_centroids.equals(self.centroids):
    #             break
    #         self.centroids = new_centroids

    # def __predict_multi_knn(self, X: np.ndarray) -> np.ndarray:
    #     distances = np.sqrt(((X - self.centroids.iloc[:, np.newaxis]) ** 2).sum(axis=2))
    #     labels = np.argmin(distances, axis=1)
    #     return labels

    def __train_multi_svm(self):
        param_grid = [
            {"C": [0.2, 0.5, 0.8, 1, 10, 100, 1000], "kernel": ["linear"]},
            {
                "C": [0.2, 0.5, 0.8, 1, 10, 100, 1000],
                "gamma": [0.01, 0.001, 0.0001, 0.00001],
                "kernel": ["rbf"],
            },
        ]

        clf = GridSearchCV(SVC(), param_grid)
        clf.fit(self.__pre_processed_train_X, self.__train_y)

        gamma: float | Literal["scale", "auto"]

        try:
            gamma = clf.best_params_["gamma"]
        except:
            gamma = "scale"

        self.__classifier_model = OneVsOneClassifier(
            SVC(
                C=clf.best_params_["C"],
                kernel=clf.best_params_["kernel"],
                degree=3,
                gamma=gamma,
                coef0=0.0,
                shrinking=True,
                probability=False,
                tol=1e-3,
            )
        )

        self.__classifier_model.fit(self.__pre_processed_train_X, self.__train_y)

    def __predict_multi_svm(self, X_test: np.ndarray):
        return self.__classifier_model.predict(X_test)

    def __train_one_vs_all_minimum_distance(self) -> None:
        data: np.ndarray = self.__pre_processed_train_X

        self.__mean_target_class = np.zeros([data.shape[1]])

        self.__mean_other_classes = np.zeros([data.shape[1]])

        for feature_index in range(data.shape[1]):
            feature_data: np.ndarray = data[:, feature_index]

            self.__mean_target_class[feature_index] = feature_data[
                self.__train_y == self.__target_class
            ].mean()

            self.__mean_other_classes[feature_index] = feature_data[
                self.__train_y != self.__target_class
            ].mean()

    def __predict_one_vs_all_minimum_distance(
        self, test_X: np.ndarray
    ) -> np.ndarray:
        distances = cdist(
            [self.__mean_target_class, self.__mean_other_classes],
            test_X,
            self.__distance_type,
        )

        min_dist = np.argmin(distances, axis=0)

        return np.asarray(
            [
                self.__target_class if min_dist[i] == 0 else "other"
                for i in range(min_dist.size)
            ]
        )

    def __train_GNB(self) -> None:
        gnb = GaussianNB()
        self.__classifier_model = gnb.fit(self.__pre_processed_train_X, self.__train_y)

    def __predict_GNB(self, test_X: np.ndarray) -> np.ndarray:
        return self.__classifier_model.predict(test_X)

    def __train_Knn(self) -> None:
        knn = KNeighborsClassifier(n_neighbors=3)
        self.__classifier_model = knn.fit(self.__pre_processed_train_X, self.__train_y)

    def __predict_Knn(self, test_X: np.ndarray) -> np.ndarray:
        return self.__classifier_model.predict(test_X)

    def get_statistics(
        self,
        target: np.ndarray,
        show_matrix: bool,
    ) -> None:
        labels = np.unique(self.__train_y)

        if labels.size == 2:
            cm = confusion_matrix(target, self.__predicted_labels, labels=labels)

            stats: dict[str, float] = dict()
            stats["sensitivity"] = float(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
            stats["specificity"] = float(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
            stats["precision"] = float(cm[0, 0] / (cm[0, 0] + cm[1, 0]))

            print(stats)

        if show_matrix:
            plt.figure()
            ConfusionMatrixDisplay.from_predictions(
                target, self.__predicted_labels, labels=labels
            )
            plt.plot()

            if self.__target_class == None:
                return

            target_normalized = np.asarray(
                [
                    1 if target[i] == self.__target_class else 0
                    for i in range(target.size)
                ]
            )

            predicted_normalized = np.asarray(
                [
                    1 if self.__predicted_labels[i] == self.__target_class else 0
                    for i in range(self.__predicted_labels.size)
                ]
            )

            plt.figure()
            RocCurveDisplay.from_predictions(
                target_normalized,
                predicted_normalized,
            )
            plt.plot()
