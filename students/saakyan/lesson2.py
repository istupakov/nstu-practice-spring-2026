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
        ost = y - self.predict(x)
        return np.mean(ost**2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        ss_res = np.sum((y - self.predict(x)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errors = y - self.predict(x)
        n = x.shape[0]
        dw = (-2 / n) * x.T @ errors
        db = (-2 / n) * np.sum(errors)
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        st = x @ self.weights + self.bias
        sigmoid = 1 / (1 + np.exp(-st))
        return sigmoid

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        pred = np.clip(self.predict(x), eps, 1 - eps)
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        pred_prob = self.predict(x)
        pred = pred_prob >= 0.5

        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        if type == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn)
        if type == "precision":
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)
        if type == "recall":
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)
        if type == "F1":
            precision = self.metric(x, y, "precision")
            recall = self.metric(x, y, "recall")
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        if type == "AUROC":
            pos = pred_prob[y == 1]
            neg = pred_prob[y == 0]

            if len(pos) == 0 or len(neg) == 0:
                return 0.5

            good_pairs = np.sum(pos[:, None] > neg[None, :])
            equal_pairs = np.sum(pos[:, None] == neg[None, :])
            return (good_pairs + 0.5 * equal_pairs) / (len(pos) * len(neg))

        return 0.0

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errors = self.predict(x) - y
        n = x.shape[0]
        dw = x.T @ errors * (1 / n)
        db = np.sum(errors) * (1 / n)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Саакян Айк Алексанович, ПМ-34"

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
        if batch_size is None or batch_size <= 0 or batch_size > x.shape[0]:
            batch_size = x.shape[0]

        for _ in range(n_iter):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                dw, db = model.grad(x_batch, y_batch)
                model.weights -= lr * dw
                model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.01, "batch_size": 8}
