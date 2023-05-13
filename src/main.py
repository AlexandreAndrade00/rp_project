import csv
from utils import read_and_standardize_data
import classifier as cl
import numpy as np

TARGET_CLASS = "blues"


def main():
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        res = [
            # "label",
            "num_features",
            "reduction_methos",
            "sensivity_mean",
            "sensivity_std",
            "specificity_mean",
            "specificity_std",
            "precision_mean",
            "precision_std",
        ]
        writer.writerow(res)

    # for label in [
    #     "blues",
    #     "classical",
    #     "country",
    #     "disco",
    #     "hiphop",
    #     "jazz",
    #     "metal",
    #     "pop",
    #     "reggae",
    #     "rock",
    # ]:
    X_train, X_test, y_train, y_test = read_and_standardize_data(False)

    for reduction in ["None", "PCA", "LDA"]:
        for i in range(0, X_test.shape[1], 3):
            try:
                model: cl.Classifier = cl.Classifier(X_train, y_train)

                model.feature_selection(i + 1)

                if reduction != "None":
                    model.feature_reduction(reduction)

                model.train("svm")
                # model.train("one_vs_all", distance_type="mahalanobis")
                # model.train("one_vs_all", distance_type="euclidean")

                model.predict(X_test)

                # stats = model.get_statistics(y_test, True).values()
                stats = model.get_statistics(y_test, False)

                # print(f"number features:{i} Feature Reduction: {reduction} Stats: {stats}")

                with open("results.csv", "a") as f:
                    # 'a' instead of 'w' makes things append instead of ovewriting
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            # label,
                            i + 1,
                            reduction,
                            stats["sensitivity"][0],
                            stats["sensitivity"][1],
                            stats["specificity"][0],
                            stats["specificity"][1],
                            stats["precision"][0],
                            stats["precision"][1],
                        ]
                    )
            except  np.linalg.LinAlgError:
                with open("results.csv", "a") as f:
                    # 'a' instead of 'w' makes things append instead of ovewriting
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            # label,
                            i + 1,
                            reduction,
                            "Null",
                            "Null",
                            "Null",
                            "Null",
                            "Null",
                            "Null",
                        ]
                    )


if __name__ == "__main__":
    main()
