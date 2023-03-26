from typing import Literal
import pandas as pd
import numpy as np
import pre_processing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Classifier:
    __train_data: pd.DataFrame
    __train_data_labels: pd.Series
    __pre_processed_train_data: pd.DataFrame | None = None
    __pre_process_method: Literal["PCA", "LDA", "KW", None] = None
    __predicted_labels: np.ndarray
    __n_components_pre_process: int | None

    def __init__(self, train_data: pd.DataFrame) -> None:
        self.__train_data = train_data.drop("label", axis=1)
        self.__train_data_labels = train_data.label

    def pre_process(self, method: str, n_components: int | None) -> None:
        self.__pre_processed_train_data = self.__pre_process(
            method, self.__train_data, n_components, self.__train_data_labels
        )

        self.__n_components_pre_process = n_components

    def __pre_process(
        self,
        method: str,
        data: pd.DataFrame,
        n_components: int | None,
        labels: pd.Series | None = None,
    ) -> pd.DataFrame:
        result: pd.DataFrame

        match method:
            case "PCA":
                if n_components is None:
                    raise TypeError("Number of components should not be None")

                result = pre_processing.comput_PCA(
                    features=data, n_components=n_components
                )

            case "LDA":
                if labels is None:
                    raise TypeError("Labels are empty")

                result = pre_processing.comput_LDA(
                    features=data, n_components=1, labels=labels
                )

            case "KW":
                if labels is None:
                    raise TypeError("Labels are empty")

                if n_components is None:
                    raise TypeError("Number of components should not be None")

                result = pre_processing.comput_kruskal(
                    features=data,
                    n_components=n_components,
                    labels_indexes=self.__get_labels_by_indexes(data, labels),
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

            case _:
                raise ValueError(model)

        self.__selected_model: str = model

    def predict(self, sample: pd.DataFrame) -> np.ndarray:
        test_data: pd.DataFrame = sample.drop("label", axis=1)
        test_labels: pd.Series = sample.label

        if self.__pre_process_method is not None:
            test_data = self.__pre_process(
                self.__pre_process_method,
                test_data,
                self.__n_components_pre_process,
                test_labels,
            )

        match self.__selected_model:
            case "one_vs_all":
                self.__predicted_labels = (
                    self.__predict_one_vs_all_euclidean_minimum_distance(test_data)
                )
            case _:
                raise NotImplementedError(self.__selected_model)

        return self.__predicted_labels

    def __train_one_vs_all_minimum_distance(self) -> None:
        data: pd.DataFrame = (
            self.__pre_processed_train_data
            if (self.__pre_processed_train_data is not None)
            else self.__train_data
        )

        classes_indexes: dict[str, pd.Index] = self.__get_labels_by_indexes(
            data, self.__train_data_labels
        )

        target_class_indexes: pd.Index = classes_indexes[self.__target_class]

        other_class_indexes: pd.Index = pd.Index([]).append(
            [
                value
                for key, value in classes_indexes.items()
                if key != self.__target_class
            ]
        )

        self.__mean_target_class = np.zeros([data.shape[1], 1])

        self.__mean_other_classes = np.zeros([data.shape[1], 1])

        if self.__distance_type == "mahalanobis":
            cov_target_class = data.loc[target_class_indexes].cov()

            cov_other_class = data.loc[other_class_indexes].cov()

            covariance = (cov_target_class + cov_other_class) / 2

            self.__cov_inv = pd.DataFrame(
                np.linalg.pinv(covariance.values), covariance.columns, covariance.index
            )

        current_feature: int = 0
        for feature in data.columns:
            feature_data: pd.DataFrame = data[feature]

            self.__mean_target_class[current_feature, 0] = feature_data[
                target_class_indexes
            ].mean()

            self.__mean_other_classes[current_feature, 0] = feature_data[
                other_class_indexes
            ].mean()

            current_feature = current_feature + 1

    def __predict_one_vs_all_euclidean_minimum_distance(
        self, sample: pd.Series | pd.DataFrame
    ) -> np.ndarray:
        def predict_one(one_sample: pd.Series):
            if self.__distance_type == "euclidean":
                return np.dot(
                    (
                        self.__mean_target_class[:, 0] - self.__mean_other_classes[:, 0]
                    ).T,
                    np.subtract(
                        one_sample,
                        0.5
                        * np.add(
                            self.__mean_target_class[:, 0],
                            self.__mean_other_classes[:, 0],
                        ),
                    ),
                )

            elif self.__distance_type == "mahalanobis":
                return np.dot(
                    np.dot(
                        np.subtract(
                            self.__mean_target_class[:, 0],
                            self.__mean_other_classes[:, 0],
                        ).T,
                        self.__cov_inv.values,
                    ),
                    np.subtract(
                        one_sample,
                        0.5
                        * np.add(
                            self.__mean_target_class[:, 0],
                            self.__mean_other_classes[:, 0],
                        ),
                    ),
                )
            else:
                raise ValueError(self.__distance_type)

        if type(sample) is pd.Series:
            result = predict_one(sample)
        elif type(sample) is pd.DataFrame:
            result = np.zeros(sample.shape[0])

            for index in range(sample.shape[0]):
                result_teste = predict_one(sample.iloc[index, :])
                result[index] = result_teste

        else:
            raise TypeError(sample)

        return np.asarray(
            [self.__target_class if elem > 0 else "other" for elem in result]
        )

    def get_statistics(
        self, target: np.ndarray, target_class_name: str
    ) -> dict[str, float]:
        cm = confusion_matrix(
            target, self.__predicted_labels, labels=[target_class_name, "other"]
        )

        stats: dict[str, float] = dict()
        stats["sensitivity"] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        stats["specificity"] = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        stats["precision"] = cm[0, 0] / (cm[0, 0] + cm[1, 0])

        print(stats)

        ConfusionMatrixDisplay.from_predictions(
            target, self.__predicted_labels, labels=[target_class_name, "other"]
        )

        plt.plot()

        return stats

    def __get_labels_by_indexes(
        self, data: pd.DataFrame, labels: pd.Series
    ) -> dict[str, pd.Index]:
        """Get the dataframe indexes splitted by labels

        Args:
            data (pd.DataFrame): data

        Returns:
            dict[str, pd.Index]: str - label; value - dataframe indexes
        """

        data_with_labels: pd.DataFrame = data.copy(deep=True)

        data_with_labels["label"] = labels

        labels_indexes: dict[str, pd.Index] = {}

        for label in data_with_labels.label.unique():
            labels_indexes[label] = data_with_labels[
                data_with_labels.label == label
            ].index

        return labels_indexes
