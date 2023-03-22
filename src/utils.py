import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def read_and_standardize_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read data from data file, split in train and test groups and standardize the data with z-score

    Returns:
        tuple[pd.DataFrame, dict[(str) Class, (pd.Index) Dataframe Indexes]]:
            Return the standardized training data and the indexes of the samples grouped by class
    """

    # read the data
    data: pd.DataFrame = pd.read_csv("../data/dados.csv")

    # split the data, 80% training, 20% test
    data_train, data_test = train_test_split(data, test_size=0.20, stratify=data.label)

    def standardize(data):
        data_df: pd.DataFrame = pd.DataFrame(data)

        # get the only the features (remove song name and label)
        features: pd.DataFrame = data_df.iloc[:, 1:-1]

        # standardize the data with z-score
        standarized_features: pd.DataFrame = pd.DataFrame(
            StandardScaler().fit_transform(features),
            columns=features.columns,
            index=features.index,
        )

        return pd.concat([standarized_features, data_df["label"]], axis=1)

    return standardize(data_train), standardize(data_test)


def get_labels_by_indexes(data: pd.DataFrame) -> dict[str, pd.Index]:
    """Get the dataframe indexes splitted by labels

    Args:
        data (pd.DataFrame): data

    Returns:
        dict[str, pd.Index]: str - label; value - dataframe indexes
    """
    labels_indexes: dict[str, pd.Index] = {}

    for label in data.label.unique():
        labels_indexes[label] = data[data.label == label].index

    return labels_indexes

def get_statistics(target: np.ndarray, predicted: np.ndarray, target_class_name: str):
    cm = confusion_matrix(target, predicted, labels=[target_class_name, "other"]) 
    
    stats= dict()
    stats["sensitivity"] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    stats["specificity"] = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    stats["precision"] = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    
    print(stats)
    
    ConfusionMatrixDisplay.from_predictions(target, predicted, labels=[target_class_name, "other"])
    plt.plot()
