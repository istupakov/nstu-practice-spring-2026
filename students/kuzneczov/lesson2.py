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
        return np.mean((y - self.predict(x)) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - np.sum((y - self.predict(x)) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        weightsGrad = -2 * ((y - self.predict(x)) @ x) / x.shape[0]
        biasGrad = -2 * np.mean(y - self.predict(x))
        return weightsGrad, biasGrad


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
        predict = self.predict(x)
        return -np.mean(y * np.log(predict) + (1 - y) * np.log(1 - predict))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        match type:
            case "accuracy":
                predict = self.predict(x)
                predict = predict >= 0.5
                return np.sum(y == predict) / y.size
            case "precision":
                predict = self.predict(x)
                predict = predict >= 0.5
                TP = np.sum((y == 1) & (predict == 1))
                FP = np.sum((y == 0) & (predict == 1))
                if TP + FP == 0:
                    return 0
                return TP / (TP + FP)
            case "recall":
                predict = self.predict(x)
                predict = predict >= 0.5
                TP = np.sum((y == 1) & (predict == 1))
                FN = np.sum((y == 1) & (predict == 0))
                if TP + FN == 0:
                    return 0
                return TP / (TP + FN)
            case "F1":
                predict = self.predict(x)
                predict = predict >= 0.5
                TP = np.sum((y == 1) & (predict == 1))
                FP = np.sum((y == 0) & (predict == 1))
                FN = np.sum((y == 1) & (predict == 0))
                if TP + FP + FN == 0:
                    return 0
                return TP / (TP + 0.5 * (FP + FN))
            case "AUROC":
                n = 10000
                h = 1 / n
                xc = np.zeros(n)
                yc = np.zeros(n)
                predict = self.predict(x)
                for i, threshold in enumerate(np.linspace(1, 0, n)):
                    threshold = 1 - i * h
                    pred = predict >= threshold
                    TP = np.sum((y == 1) & (pred == 1))
                    FN = np.sum((y == 1) & (pred == 0))
                    FP = np.sum((y == 0) & (pred == 1))
                    TN = np.sum((y == 0) & (pred == 0))
                    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                    xc[i] = FPR
                    yc[i] = TPR
                return np.trapezoid(yc, xc)
        return 0.0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        predict = self.predict(x)
        biasGrad = (np.sum(predict - y)) / y.size
        weightsGrad = ((predict - y) @ x) / y.size
        return weightsGrad, biasGrad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кузнецов Александр Павлович, ПМ-34"

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
            batch_size = x.shape[0]
        for _i in range(n_epoch):
            for _j in range(0, x.shape[0], batch_size):
                weightsGrad, biasGrad = model.grad(x[_j : _j + batch_size], y[_j : _j + batch_size])
                model.weights -= weightsGrad * lr
                model.bias -= biasGrad * lr

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.1, "batch_size": 7}
