from utils import read_and_standardize_data
import classifier as cl

TARGET_CLASS = "blues"


def main():
    X_train, X_test, y_train, y_test = read_and_standardize_data(True, "blues")

    model: cl.Classifier = cl.Classifier(X_train, y_train)

    model.feature_selection()

    model.feature_reduction("LDA")

    model.train("one_vs_all", distance_type="euclidean")

    model.predict(X_test)

    model.get_statistics(y_test, True)

    

if __name__ == "__main__":
    main()
