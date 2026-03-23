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
        return float(np.mean((self.predict(x) - y) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:

        y_hat = self.predict(x)
        rss = np.sum((y - y_hat) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        return float(1 - rss / (tss + 1e-12))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        error = self.predict(x) - y
        dw = (2 / n) * (x.T @ error)
        db = 2 * np.mean(error)
        return dw, np.array(db)


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

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = (self.predict(x) >= 0.5).astype(int)
        return float(np.mean(y_hat == y))

    def precision(self, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = (self.predict(x) >= 0.5).astype(int)
        tp = np.sum((y_hat == 1) & (y == 1))
        fp = np.sum((y_hat == 1) & (y == 0))
        return float(tp / (tp + fp + 1e-15))

    def recall(self, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = (self.predict(x) >= 0.5).astype(int)
        tp = np.sum((y_hat == 1) & (y == 1))
        fn = np.sum((y_hat == 0) & (y == 1))
        return float(tp / (tp + fn + 1e-15))

    def f1_score(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.precision(x, y)
        r = self.recall(x, y)
        return float(2 * p * r / (p + r + 1e-15))

    def auroc(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict(x)
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        if pos_count == 0 or neg_count == 0:
            return 0.5
        ranked_indices = np.argsort(probs)
        rank_sum_pos = np.sum(np.where(y[ranked_indices] == 1)[0] + 1)
        return float((rank_sum_pos - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:
        if type == "precision":
            return self.precision(x, y)
        if type == "recall":
            return self.recall(x, y)
        if type == "F1":
            return self.f1_score(x, y)
        if type == "AUROC":
            return self.auroc(x, y)
        if type == "accuracy":
            return self.accuracy(x, y)

        return self.accuracy(x, y)

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        error = self.predict(x) - y
        dw = (x.T @ error) / n
        db = np.mean(error)
        return dw, np.array(db)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Дегтярев Кирилл Романович, ПМ-35"

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
        bs = n if batch_size is None else batch_size

        for _ in range(n_iter):
            for i in range(0, n, bs):
                x_batch = x[i : i + bs]
                y_batch = y[i : i + bs]
                if len(x_batch) == 0:
                    continue

                dw, db = model.grad(x_batch, y_batch)
                model.weights -= lr * dw
                model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.01, "batch_size": 4}
