import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_and_standardize_data(target_class: str, two_classes: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read data from data file, split in train and test groups and standardize the data with z-score

    Returns:
        tuple[pd.DataFrame, dict[(str) Class, (pd.Index) Dataframe Indexes]]:
            Return the standardized training data and the indexes of the samples grouped by class
    """

    # read the data
    data: pd.DataFrame = pd.read_csv("data/dados.csv")

    data_y = data['label'].values
    data_X = data.drop(columns=['label', 'filename']).values
    

    # split the data, 80% training, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.2, stratify=data.label, random_state=42
    )

    def standardize(data):
        return StandardScaler().fit_transform(data)
    
    if two_classes:
        y_train[y_train != target_class] = "other"
        y_test[y_test != target_class] = "other"


    return standardize(X_train), standardize(X_test), y_train, y_test
