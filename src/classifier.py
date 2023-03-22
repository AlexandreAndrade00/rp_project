import pandas as pd
import numpy as np
from utils import get_labels_by_indexes


class Classifier:
    def train(self, model: str, data: pd.DataFrame, **kwargs):
        match model:
            case "one_vs_all_1":
                self.__train_one_vs_all_euclidean_minimum_distance(
                    data, kwargs["target_class"]
                )
            case _:
                raise ValueError(model)

        self.__selected_model: str = model
        self.__target_class = kwargs["target_class"]

    def predict(self, sample: pd.Series | pd.DataFrame):
        match self.__selected_model:
            case "one_vs_all_1":
                return self.__predict_one_vs_all_euclidean_minimum_distance(sample)
            case _:
                raise NotImplementedError(self.__selected_model)

    def __train_one_vs_all_euclidean_minimum_distance(
        self, data: pd.DataFrame, target_class: str
    ):
        classes_indexes: dict[str, pd.Index] = get_labels_by_indexes(data)

        target_class_indexes: pd.Index = classes_indexes[target_class]

        other_class_indexes: pd.Index = pd.Index([]).append(
            [value for key, value in classes_indexes.items() if key != target_class]
        )

        self.__mean_target_class = np.zeros([data.shape[1] - 1, 1])

        self.__mean_other_classes = np.zeros([data.shape[1] - 1, 1])

        current_feature: int = 0
        for feature in data.columns:
            if feature == "label":
                continue

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
            return np.dot(
                np.subtract(
                    self.__mean_target_class[:, 0], self.__mean_other_classes[:, 0]
                ).T,
                one_sample
                - 0.5
                * np.add(
                    self.__mean_target_class[:, 0], self.__mean_other_classes[:, 0]
                ),
            )

        if type(sample) is pd.Series:
            result = predict_one(sample)
        elif type(sample) is pd.DataFrame:
            result = np.zeros(sample.shape[0])

            for index in range(sample.shape[0]):
                result[index] = predict_one(sample.iloc[index, :])

        else:
            raise TypeError(sample)

        return np.asarray([self.__target_class if elem > 0 else "other" for elem in result])
