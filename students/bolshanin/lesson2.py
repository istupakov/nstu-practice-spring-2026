import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_prediction = self.predict(x)
        return float(np.mean((y - y_prediction) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_prediction = self.predict(x)
        ss_res = np.sum((y - y_prediction) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0
        r_squared = 1 - ss_res / ss_tot
        return float(r_squared)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_prediction = self.predict(x)
        error = y_prediction - y
        n = len(y)
        grad_weights = (2 / n) * (x.T @ error)
        grad_bias = (2 / n) * np.sum(error)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_prediction = self.predict(x)
        eps = np.finfo(float).eps
        y_prediction = np.clip(y_prediction, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_prediction) + (1 - y) * np.log(1 - y_prediction)))

    def metric(self, x: np.ndarray, y: np.ndarray, metric_type: str = "accuracy") -> float:

        if metric_type == "accuracy":
            y_prediction = (self.predict(x) >= 0.5).astype(int)
            return float(np.mean(y_prediction == y))
        elif metric_type == "precision":
            return self._precision(x, y)
        elif metric_type == "recall":
            return self._recall(x, y)
        elif metric_type == "f1" or metric_type == "F1":
            return self._f1(x, y)
        elif metric_type == "auroc" or metric_type == "AUROC":
            return self._auroc(x, y)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_prediction = self.predict(x)
        error = y_prediction - y
        n = len(y)
        grad_weights = (1 / n) * (x.T @ error)
        grad_bias = (1 / n) * np.sum(error)
        return grad_weights, grad_bias

    def _precision(self, x: np.ndarray, y: np.ndarray) -> float:
        """Точность: TP / (TP + FP)"""
        y_pred = (self.predict(x) >= 0.5).astype(int)
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        return tp / (tp + fp + 1e-15)

    def _recall(self, x: np.ndarray, y: np.ndarray) -> float:
        """Полнота: TP / (TP + FN)"""
        y_pred = (self.predict(x) >= 0.5).astype(int)
        tp = np.sum((y_pred == 1) & (y == 1))
        fn = np.sum((y_pred == 0) & (y == 1))
        return tp / (tp + fn + 1e-15)

    def _f1(self, x: np.ndarray, y: np.ndarray) -> float:
        """F1-мера: 2 * precision * recall / (precision + recall)"""
        p = self._precision(x, y)
        r = self._recall(x, y)
        return 2 * p * r / (p + r + 1e-15)

    def _auroc(self, x: np.ndarray, y: np.ndarray) -> float:

        y_score = self.predict(x)

        order = np.argsort(y_score)
        y_sorted = y[order]

        n_pos = np.sum(y_sorted == 1)
        n_neg = np.sum(y_sorted == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        rank_sum = 0
        for i, label in enumerate(y_sorted):
            if label == 1:
                rank_sum += i + 1

        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        return float(auc)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Большанин Егор Андреевич, ПМ-33"

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
        n_iter: int,
        batch_size: int | None = None,
    ) -> None:
        n = x.shape[0]

        for _ in range(n_iter):
            if batch_size is None:
                grad_w, grad_b = model.grad(x, y)
                model.weights -= lr * grad_w
                model.bias -= lr * grad_b
            else:
                for i in range(0, n, batch_size):
                    end = min(i + batch_size, n)
                    x_batch = x[i:end]
                    y_batch = y[i:end]

                    grad_w, grad_b = model.grad(x_batch, y_batch)
                    model.weights -= lr * grad_w
                    model.bias -= lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict:

        return {"lr": 0.002, "batch_size": 1, "n_iter": 25}
