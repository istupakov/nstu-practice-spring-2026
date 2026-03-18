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
        y_pred = self.predict(x)
        return float(np.mean((y - y_pred) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(1 - self.loss(x, y) / np.var(y))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(x)
        dw = -2 * x.T @ (y - y_pred) / x.shape[0]
        db = -2 * np.mean(y - y_pred)
        return dw, db


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
        p = np.clip(self.predict(x), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        p = self.predict(x)
        y_pred = (p >= 0.5).astype(int)

        if type == "accuracy":
            return float(np.mean(y_pred == y))

        tp = float(np.sum((y_pred == 1) & (y == 1)))
        # tn = float(np.sum((y_pred == 0) & (y == 0)))
        fp = float(np.sum((y_pred == 1) & (y == 0)))
        fn = float(np.sum((y_pred == 0) & (y == 1)))

        if type == "precision":
            return tp / (tp + fp) if tp + fp > 0 else 0.0
        elif type == "recall":
            return tp / (tp + fn) if tp + fn > 0 else 0.0
        elif type == "F1":
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        elif type == "AUROC":
            pos_scores = p[y == 1]
            neg_scores = p[y == 0]
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                return 0.5
            correct_pairs = np.sum(pos_scores[:, np.newaxis] > neg_scores[np.newaxis, :])
            tie_pairs = np.sum(pos_scores[:, np.newaxis] == neg_scores[np.newaxis, :])
            return float((correct_pairs + 0.5 * tie_pairs) / (len(pos_scores) * len(neg_scores)))
        return 0.0

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        dw = x.T @ (p - y) / x.shape[0]
        db = np.mean(p - y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кудрявцев Павел Павлович, ПМ-35"

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
        if batch_size is None:
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
        return {"lr": 0.08, "batch_size": 1}
