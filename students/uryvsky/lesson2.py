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
        return np.mean(np.square(y - self.predict(x)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (self.loss(x, y) / np.var(y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        dw = -2.0 * x.T @ (y - self.predict(x)) / x.shape[0]
        db = -2.0 * np.sum(y - self.predict(x)) / x.shape[0]
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
        eps = 1e-15
        p = np.clip(self.predict(x), eps, 1 - eps)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        pred = self.predict(x)
        solution_pred = pred >= 0.5

        TP = np.sum((solution_pred == 1) & (y == 1))
        TN = np.sum((solution_pred == 0) & (y == 0))
        FP = np.sum((solution_pred == 1) & (y == 0))
        FN = np.sum((solution_pred == 0) & (y == 1))

        if type == "accuracy":
            return ((TP + TN) / (TP + FP + TN + FN)) if (TP + FP + TN + FN) != 0 else 0.0
        elif type == "precision":
            return (TP / (TP + FP)) if (TP + FP) != 0 else 0.0
        elif type == "recall":
            return (TP / (TP + FN)) if (TP + FN) != 0 else 0.0
        elif type == "F1":
            return (TP / (TP + 0.5 * (FP + FN))) if (TP + 0.5 * (FP + FN)) != 0 else 0.0
        else:
            x_arr = []
            y_arr = []

            P = np.sum(y == 1)
            N = np.sum(y == 0)

            if P == 0 or N == 0:
                return 0.5

            for segment in np.linspace(1.0, 0.0, 1000):
                TP_ = np.sum(pred[y == 1] >= segment)
                FP_ = np.sum(pred[y == 0] >= segment)

                TPR = TP_ / P
                FPR = FP_ / N

                x_arr.append(FPR)
                y_arr.append(TPR)

            return np.trapezoid(y_arr, x_arr)
        return 0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        dw = (x.T @ (self.predict(x) - y)) / len(y)
        db = np.mean(self.predict(x) - y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Урывский Александр Александрович, ПМ-31"

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
                for i in range(0, len(x), batch_size):
                    dw, db = model.grad(x[i : i + batch_size], y[i : i + batch_size])
                    model.weights -= lr * dw
                    model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.0001, "batch_size": 25}
