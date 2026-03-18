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
        return np.sum(np.square(y - self.predict(x))) / (y.size)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (np.sum(np.square(y - self.predict(x))) / np.sum(np.square(y - np.sum(y) / y.size)))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        db = -2 * np.sum(y - self.predict(x)) / y.size
        dw = -2 * x.T @ (y - self.predict(x)) / y.size
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.bias + x @ self.weights)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return -np.sum(y * np.log(self.predict(x)) + (1 - y) * np.log(1 - self.predict(x))) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray, str="accuracy") -> float:
        doorstep = 0.5
        pred = self.predict(x)
        pred1 = self.predict(x) >= doorstep
        TP = np.sum((pred1 == 1) & (y == 1))
        FP = np.sum((pred1 == 1) & (y == 0))
        FN = np.sum((pred1 == 0) & (y == 1))
        TN = np.sum((pred1 == 0) & (y == 0))
        if str == "accuracy":
            metric = (TP + TN) / (TP + FP + TN + FN) if ((TP + FP + TN + FN) != 0) else 0
        if str == "precision":
            metric = TP / (TP + FP) if ((TP + FP) != 0) else 0
        if str == "recall":
            metric = TP / (TP + FN) if ((TP + FN) != 0) else 0
        if str == "F1":
            metric = TP / (TP + (FP + FN) / 2) if ((TP + (FP + FN) / 2) != 0) else 0
        if str == "AUROC":
            pos = pred[y == 1]
            neg = pred[y == 0]
            score = pos[:, None] > neg[None, :]
            metric = np.sum(score) / (len(pos) * len(neg)) if (len(pos) * len(neg) != 0) else 0

        return metric

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        db = np.sum(self.predict(x) - y) / y.size
        dw = (x.T @ (self.predict(x) - y)) / len(y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Воробьев Никита Александрович, ПМ-31"

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
        if batch_size is None or batch_size <= 0:
            for _ in range(n_epoch):
                dw, db = model.grad(x, y)
                model.weights -= lr * dw
                model.bias -= lr * db

        else:
            for _ in range(n_epoch):
                for i in range(0, x.shape[0], batch_size):
                    x1 = x[i : i + batch_size]
                    y1 = y[i : i + batch_size]
                    dw, db = model.grad(x1, y1)
                    model.weights -= lr * dw
                    model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.001, "batch_size": 25}
