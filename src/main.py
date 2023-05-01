from utils import read_and_standardize_data
import classifier as cl

TARGET_CLASS = "blues"


def main():

    X_train, X_test, y_train, y_test = read_and_standardize_data(TARGET_CLASS, True)
    
    def run_one_vs_all(distance_type:str ):

        file = open("stats.txt", "w")

        model: cl.Classifier = cl.Classifier(X_train, y_train)

        model.pre_process("LDA")

        model.train("one_vs_all", target_class=TARGET_CLASS, distance_type=distance_type)

        model.predict(X_test, y_test)

        model.get_statistics(y_test, TARGET_CLASS, False, file=file)

        file.flush()
        file.close()

    run_one_vs_all("euclidean")


if __name__ == "__main__":
    main()
