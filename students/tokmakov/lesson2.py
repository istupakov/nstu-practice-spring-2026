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

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        y_pred = self.predict(x)
        errors = y - y_pred
        grad_w = -2 * x.T @ errors / x.shape[0]
        grad_b = float(-2 * np.mean(errors))
        return grad_w, grad_b


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

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:
        p = self.predict(x)
        y_pred = (p >= 0.5).astype(int)

        if type is None or type == "accuracy":
            return float(np.mean(y_pred == y))

        tp = float(np.sum((y_pred == 1) & (y == 1)))
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

            correct_pairs = np.sum(pos_scores[:, None] > neg_scores[None, :])
            tie_pairs = np.sum(pos_scores[:, None] == neg_scores[None, :])

            return float((correct_pairs + 0.5 * tie_pairs) / (len(pos_scores) * len(neg_scores)))

        return 0.0

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        p = self.predict(x)
        grad_w = x.T @ (p - y) / x.shape[0]
        grad_b = float(np.mean(p - y))
        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Токмаков Дмитрий Евгеньевич, ПМ-31"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng=rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng=rng or np.random.default_rng())

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
            batch_size = int(x.shape[0])

        for _ in range(n_iter):
            for i in range(x.shape[0] // batch_size):
                x_batch = x[i * batch_size : (i + 1) * batch_size]
                y_batch = y[i * batch_size : (i + 1) * batch_size]
                dw, db = model.grad(x_batch, y_batch)
                model.weights -= lr * dw
                model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.08, "batch_size": 32}
