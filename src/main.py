from utils import read_and_standardize_data
import classifier as cl
import numpy as np
TARGET_CLASS = "blues"


def main():
    X_train, X_test, y_train, y_test = read_and_standardize_data(False)
    # True means one vs all + specify target class, False means one vs one

    model: cl.Classifier = cl.Classifier(X_train, y_train)

    model.feature_selection()

    model.feature_reduction("LDA")

    model.train("knn")

    model.predict(X_test)

    model.get_statistics(y_test, True)

    print(np.unique(y_train))

    

if __name__ == "__main__":
    main()
