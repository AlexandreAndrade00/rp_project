from utils import read_and_standardize_data
import classifier as cl

TARGET_CLASS = "rock"


def main():

    standarized_train, standarized_test = read_and_standardize_data(TARGET_CLASS)

    print("Euclidean\n")
    run_one_vs_all(standarized_train, standarized_test, "euclidean")
    run_one_vs_all(standarized_train, standarized_test, "euclidean", pre_process_method="PCA", n_components=13)
    run_one_vs_all(standarized_train, standarized_test, "euclidean", pre_process_method="KW", n_components=6)
    run_one_vs_all(standarized_train, standarized_test, "euclidean", pre_process_method="LDA")

    print("\nMahalanobis\n")
    run_one_vs_all(standarized_train, standarized_test, "mahalanobis")
    run_one_vs_all(standarized_train, standarized_test, "mahalanobis", pre_process_method="PCA", n_components=90)
    run_one_vs_all(standarized_train, standarized_test, "mahalanobis", pre_process_method="KW", n_components=15)
    run_one_vs_all(standarized_train, standarized_test, "mahalanobis", pre_process_method="LDA")



def run_one_vs_all(train_data, test_data, distance_type: str = "euclidean", pre_process_method: str | None = None, n_components: int | None = None):
    model: cl.Classifier = cl.Classifier(train_data)

    if pre_process_method is not None:
        if n_components is None:
            n_components = 10

        model.pre_process(pre_process_method, n_components)

    model.train("one_vs_all", target_class=TARGET_CLASS, distance_type=distance_type)

    model.predict(test_data)

    target_labels = test_data.iloc[:, -1].to_numpy()

    model.get_statistics(target_labels, TARGET_CLASS)


if __name__ == "__main__":
    main()
