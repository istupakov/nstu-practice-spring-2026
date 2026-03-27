import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)  # смещение

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.square(y - self.predict(x)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - self.loss(x, y) / np.var(y)  # / дисперсию

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        dw = (-2 / len(x)) * np.dot(x.T, (y - p))
        db = -2 * np.mean(y - p)
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        threshold = 0.5
        p = self.predict(x)

        TP = (p[y == 1] >= threshold).sum()
        FP = (p[y == 0] >= threshold).sum()
        TN = (p[y == 0] < threshold).sum()
        FN = (p[y == 1] < threshold).sum()

        if type == "accuracy":
            return np.mean((p >= threshold) == y)
        if type == "precision":
            if TP + FP == 0:
                return 0.0
            return TP / (TP + FP)
        if type == "recall":
            if TP + FN == 0:
                return 0.0
            return TP / (TP + FN)
        if type == "F1":
            precision = self.metric(x, y, "precision")
            recall = self.metric(x, y, "recall")
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        else:  # type == Auroc
            x_arr = []
            y_arr = []
            P = TP + FN
            N = FP + TN
            for threshold in np.linspace(1.0, 0.0, 1000):
                TP = (p[y == 1] >= threshold).sum()
                FP = (p[y == 0] >= threshold).sum()
                TPR = TP / P
                FPR = FP / N
                x_arr.append(FPR)
                y_arr.append(TPR)
            return np.trapezoid(y_arr, x_arr)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        dw = 1 / len(x) * np.dot(x.T, (p - y))
        db = np.mean(p - y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кузьмин Александр Андреевич, ПМ-35"

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
        if batch_size is None:
            batch_size = len(x)
        for _ in range(n_epoch):
            for i in range(0, len(x), batch_size):
                dw, db = model.grad(x[i : i + batch_size], y[i : i + batch_size])
                model.weights -= lr * dw
                model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.0003, "batch_size": 1}
