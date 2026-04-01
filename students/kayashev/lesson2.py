import numpy as np


class LinearRegression:
    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.square(self.predict(x) - y)) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        return 1 - np.sum((y - prediction) ** 2) / np.sum((y - np.average(y)) ** 2)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        return -2 / y.size * (y - prediction) @ x, -2 * np.mean(y - prediction)


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1 / (1 + (np.exp(-z)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        predict = self.predict(x)
        return -np.sum(y * np.log(predict) + (-y + 1) * np.log(-predict + 1)) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        def round(x, alpha):
            return 1 if (x >= alpha) else 0

        vround = np.vectorize(round)
        predict = self.predict(x)
        roundedpredict = vround(predict, 0.5)
        roundedy = vround(y, 0.5)
        tp = np.sum(np.logical_and(roundedpredict, roundedy).astype(int))
        tn = np.sum(np.logical_and(np.logical_not(roundedpredict), np.logical_not(roundedy)).astype(int))
        fp = np.sum(np.logical_and(roundedpredict, np.logical_not(roundedy)).astype(int))
        fn = y.size - tp - tn - fp
        if type == "accuracy":
            metric = (tp + tn) / y.size
        if type == "precision":
            metric = tp / (tp + fp) if tp + fp != 0 else 0
        if type == "recall":
            metric = tp / (tp + fn) if tp + fn != 0 else 0
        if type == "F1":
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            metric = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        if type == "AUROC":
            xarr = []
            yarr = []
            for alpha in np.linspace(1.0, 0.0, 1000):
                roundedpredict = vround(predict, alpha)
                roundedy = vround(y, alpha)
                tp = np.sum(np.logical_and(roundedpredict, roundedy).astype(int))
                tn = np.sum(np.logical_and(np.logical_not(roundedpredict), np.logical_not(roundedy)).astype(int))
                fp = np.sum(np.logical_and(roundedpredict, np.logical_not(roundedy)).astype(int))
                fn = y.size - tp - tn - fp
                tpr = tp / (tp + fn) if tp + fn != 0 else 0
                fpr = fp / (fp + tn) if fp + tn != 0 else 0
                xarr.append(fpr)
                yarr.append(tpr)
            metric = 1 + np.trapezoid(yarr, xarr)
        return metric

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        predict = self.predict(x)
        return -x.T @ (y - predict) / y.size, -np.mean(y - predict)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Каяшев Валентин Константинович, ПМ-31"

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
            for _ in range(n_epoch):
                dw, db = model.grad(x, y)
                model.weights -= lr * dw
                model.bias -= lr * db
        else:
            for _ in range(n_epoch):
                for i in range(0, y.size, batch_size):
                    dw, db = model.grad(x[i : i + batch_size], y[i : i + batch_size])
                    model.weights -= lr * dw
                    model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.42, "batch_size": 42}
