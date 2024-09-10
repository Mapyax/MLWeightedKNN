import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def weighted_knn(x_train, y_train, x_test, k, q):
    predictions = []
    for test_point in x_test:
        distances = np.linalg.norm(x_train - test_point, axis=1)
        sorted_indices = np.argsort(distances)

        k_nearest_neighbors = sorted_indices[:k]
        neighbor_classes = y_train[k_nearest_neighbors]

        weights = np.array([q**i for i in range(k)])
        class_votes = Counter()

        for idx, neighbor_class in enumerate(neighbor_classes):
            class_votes[neighbor_class] += weights[idx]

        predicted_class = class_votes.most_common(1)[0][0]
        predictions.append(predicted_class)

    return np.array(predictions)


def compute_accuracy_for_k_q(x_train, y_train, x_test, y_test, k, q):
    y_pred_train = weighted_knn(x_train, y_train, x_train, k, q)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    y_pred = weighted_knn(x_train, y_train, x_test, k, q)
    accuracy_test = accuracy_score(y_test, y_pred)
    return {
        "k": k,
        "q": q,
        "Точность обучающей": accuracy_train,
        "Точность тестовой": accuracy_test,
    }


def main():
    df = pd.read_csv("data1.csv")

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    x_class_0 = x[y == 0]
    x_class_1 = x[y == 1]

    min_class_size = min(len(x_class_0), len(x_class_1))

    x_class_0_balanced = np.random.choice(len(x_class_0), min_class_size, replace=False)
    x_class_1_balanced = np.random.choice(len(x_class_1), min_class_size, replace=False)

    x_balanced = np.vstack(
        (x_class_0[x_class_0_balanced], x_class_1[x_class_1_balanced])
    )
    y_balanced = np.array([0] * min_class_size + [1] * min_class_size)

    results = []
    for _ in range(10):
        x_train, x_test, y_train, y_test = train_test_split(
            x_balanced, y_balanced, test_size=0.33
        )
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    compute_accuracy_for_k_q,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    k,
                    q_iter / 10,
                )
                for k in range(1, 8)
                for q_iter in range(1, 10)
            ]
            for future in futures:
                results.append(future.result())

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Точность тестовой", ascending=False)
    print(results_df)
    results_df.to_excel("accuracies.xlsx", index=False)


if __name__ == "__main__":
    main()
