from io import TextIOWrapper
from typing import Literal
import pandas as pd
import numpy as np
import pre_processing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Classifier:
    __pre_process_model: PCA | LinearDiscriminantAnalysis | None
    __train_X: np.ndarray
    __train_y: np.ndarray
    __pre_processed_train_X: np.ndarray | None = None
    __pre_process_method: Literal["PCA", "LDA", "KW", None] = None
    __predicted_labels: np.ndarray

    def __init__(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        self.__train_X = train_X
        self.__train_y = train_y
        self.__pre_process_model = None

    def pre_process(self, method: str) -> None:
        self.__pre_processed_train_X = self.__pre_process(
            method, X=self.__train_X, y=self.__train_y
        )

    def __pre_process(
        self,
        method: str,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        result: np.ndarray

        match method:
            case "PCA":
                self.__pre_process_model, result = pre_processing.comput_PCA(
                    features=X, model=self.__pre_process_model  # type: ignore
                )

            case "LDA":
                if y is None:
                    raise TypeError("Labels are empty")

                self.__pre_process_model, result = pre_processing.comput_LDA(data_X=X, data_y=y, model=self.__pre_process_model)  # type: ignore

            case "KW":
                if y is None:
                    raise TypeError("Labels are empty")

                result = pre_processing.comput_kruskal(
                    X=X,
                    y=y,
                )

            case _:
                raise ValueError("Unknown pre process method")

        self.__pre_process_method = method

        return result

    def train(self, model: str, **kwargs) -> None:
        match model:
            case "one_vs_all":
                self.__target_class = kwargs["target_class"]
                self.__distance_type = kwargs["distance_type"]

                self.__train_one_vs_all_minimum_distance()
            case "all_vs_all":
                self.__target_class = kwargs["target_class"]
                self.__distance_type = kwargs["distance_type"]



            case _:
                raise ValueError(model)

        self.__selected_model: str = model

    def predict(self, test_X: np.ndarray, test_y: np.ndarray) -> np.ndarray:
        test_data = test_X

        if self.__pre_process_method is not None:
            test_data = self.__pre_process(
                self.__pre_process_method,
                test_X,
                test_y,
            )

        match self.__selected_model:
            case "one_vs_all":
                self.__predicted_labels = (
                    self.__predict_one_vs_all_euclidean_minimum_distance(test_data)
                )

            case "all_vs_all":
                self.__predicted_labels = np.asarray([1])

            case _:
                raise NotImplementedError(self.__selected_model)

        return self.__predicted_labels

    def __train_one_vs_all_minimum_distance(self) -> None:
        data: np.ndarray = (
            self.__pre_processed_train_X
            if (self.__pre_processed_train_X is not None)
            else self.__train_X
        )

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

    def __train_all_vs_all_GNB(self) -> None:
        # TODO
        return

    def __predict_one_vs_all_euclidean_minimum_distance(
        self, test_X: np.ndarray
    ) -> np.ndarray:
        def predict_one(one_sample: np.ndarray):
            distances = cdist(
                [self.__mean_target_class, self.__mean_other_classes],
                [one_sample],
                self.__distance_type,
            )

            min_dist = np.argmin(distances)

            return min_dist

        if test_X.shape[1] == 0:
            result = predict_one(test_X)
        else:
            result = np.zeros(test_X.shape[0])

            for index in range(test_X.shape[0]):
                result_teste = predict_one(test_X[index, :])
                result[index] = result_teste

        return np.asarray(
            [self.__target_class if elem == 0 else "other" for elem in result]
        )

    def get_statistics(
        self,
        target: np.ndarray,
        target_class_name: str,
        show_matrix: bool,
        file: TextIOWrapper | None = None,
    ) -> dict[str, float]:
        cm = confusion_matrix(
            target, self.__predicted_labels, labels=[target_class_name, "other"]
        )

        stats: dict[str, float] = dict()
        stats["sensitivity"] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        stats["specificity"] = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        stats["precision"] = cm[0, 0] / (cm[0, 0] + cm[1, 0])

        print(stats)

        if file is not None:
            file.write(str(stats) + "\n")

        if show_matrix:
            ConfusionMatrixDisplay.from_predictions(
                target, self.__predicted_labels, labels=[target_class_name, "other"]
            )

            plt.plot()

        return stats
