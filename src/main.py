from utils import read_and_standardize_data, get_statistics
import classifier as cl


def main():
    TARGET_CLASS = "rock"

    standarized_train, standarized_test = read_and_standardize_data()

    model: cl.Classifier = cl.Classifier()

    model.train("one_vs_all_1", standarized_train, target_class=TARGET_CLASS)

    pred_labels = model.predict(standarized_test.iloc[:, :-1])

    target_labels = standarized_test.iloc[:, -1].to_numpy()

    target_labels[target_labels != TARGET_CLASS] = "other"

    get_statistics(target_labels, pred_labels, TARGET_CLASS)


    # print(comput_PCA(standarized_train))

    # print(comput_kruskal(standarized_train))


if __name__ == "__main__":
    main()
