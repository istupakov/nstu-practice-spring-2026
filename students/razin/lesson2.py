import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        predictions = self.predict(x)

        return np.sum((predictions - y) ** 2) / n

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(x)
        sumsq_res = np.sum((y - predictions) ** 2)

        y_mean = np.sum(y) / len(y)
        sumsq_total = np.sum((y - y_mean) ** 2)

        return 1 - sumsq_res / sumsq_total

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        predictions = self.predict(x)
        error = predictions - y

        grad_weights = (2 / n) * x.T.dot(error)
        grad_bias = (2 / n) * np.sum(error)

        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x.dot(self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        predictions = self.predict(x)

        eps = 1e-15
        predictions = np.where(predictions == 0, eps, predictions)
        predictions = np.where(predictions == 1, 1 - eps, predictions)

        term1 = y * np.log(predictions)
        term2 = (1 - y) * np.log(1 - predictions)
        loss_value = -np.sum(term1 + term2) / n

        return loss_value

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        predictions = self.predict(x)
        pred_classes = predictions >= 0.5

        # Матрица ошибок
        tp = np.sum((pred_classes == 1) & (y == 1))
        fp = np.sum((pred_classes == 1) & (y == 0))
        tn = np.sum((pred_classes == 0) & (y == 0))
        fn = np.sum((pred_classes == 0) & (y == 1))

        if type == "accuracy":
            # (TP + TN) / (TP + TN + FP + FN)
            return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        elif type == "precision":
            # TP / (TP + FP)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        elif type == "recall":
            # TP / (TP + FN)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        elif type == "F1":
            # 2 * (precision * recall) / (precision + recall)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        elif type == "AUROC":
            # Разделяем на положительные и отрицательные
            pos_scores = predictions[y == 1]
            neg_scores = predictions[y == 0]

            n_pos = len(pos_scores)
            n_neg = len(neg_scores)

            if n_pos == 0 or n_neg == 0:
                return 0.5

            # Считаем количество пар, где положительный > отрицательный
            pos_scores_col = pos_scores[:, np.newaxis]  # столбец
            neg_scores_row = neg_scores[np.newaxis, :]  # строка

            correct_pairs = np.sum(pos_scores_col > neg_scores_row)
            tie_pairs = np.sum(pos_scores_col == neg_scores_row)

            auroc = (correct_pairs + 0.5 * tie_pairs) / (n_pos * n_neg)
            return auroc

        else:
            raise ValueError(f"Unknown metric type: {type}")

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        predictions = self.predict(x)
        error = predictions - y

        grad_weights = (1 / n) * x.T.dot(error)
        grad_bias = (1 / n) * np.sum(error)

        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Разин Игорь Дмитриевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def fit(
        model: LinearRegression | LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_epoch: int,
        batch_size: int | None = None,
    ) -> None:

        n_samples = x.shape[0]

        for _ in range(n_epoch):
            # градиентный спуск
            if batch_size is None:
                grad_w, grad_b = model.grad(x, y)
                model.weights -= lr * grad_w
                model.bias -= lr * grad_b

            # Градиентный спуск с батчами
            else:
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size

                    x_batch = x[start:end]
                    y_batch = y[start:end]

                    grad_w, grad_b = model.grad(x_batch, y_batch)

                    model.weights -= lr * grad_w
                    model.bias -= lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.009, "batch_size": 4}
