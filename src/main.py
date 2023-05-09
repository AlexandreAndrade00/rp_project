from utils import read_and_standardize_data
import classifier as cl

TARGET_CLASS = "blues"


def main():
    X_train, X_test, y_train, y_test = read_and_standardize_data(False)

    model: cl.Classifier = cl.Classifier(X_train, y_train)

    model.pre_process("KW")

    model.train("gnb")

    model.predict(X_test, y_test)

    model.get_statistics(y_test, True)

    

if __name__ == "__main__":
    main()
