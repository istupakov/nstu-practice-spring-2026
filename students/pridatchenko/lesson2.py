# from matplotlib.pylab import permutation
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
        return np.sum((y - self.predict(x)) ** 2) / (y.size)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        ssres: float = np.sum((y - self.predict(x)) ** 2)
        sstot: float = np.sum((y - np.mean(y)) ** 2)

        return 1 - ssres / sstot

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n: float = y.size
        return -2 / n * (x.T @ (y - self.predict(x))), -2 / n * np.sum(y - self.predict(x))


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
        p = self.predict(x)
        return -1 / y.size * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        pred = self.predict(x)

        TP, TN, FP, FN = 0, 0, 0, 0

        y_pred = np.where(pred >= 0.5, 1, 0)

        TP = np.sum((y_pred == 1) & (y == 1))
        TN = np.sum((y_pred == 0) & (y == 0))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))

        match type:
            case "precision":
                return TP / (TP + FP) if (TP + FP) > 0 else 0.0
            case "recall":
                return TP / (TP + FN) if (TP + FN) > 0 else 0.0
            case "F1":
                return TP / (TP + 1 / 2 * (FP + FN)) if (TP + FP + FN) > 0 else 0.0
            case "accuracy":
                return (TP + TN) / y.size
            case "AUROC":
                steps = 10000
                TPR = np.zeros(steps)
                FPR = np.zeros(steps)

                threshold = 0
                for i, threshold in enumerate(np.linspace(1, 0, steps)):
                    y_pred = np.where(pred >= threshold, 1, 0)

                    TP = np.sum((y_pred == 1) & (y == 1))
                    TN = np.sum((y_pred == 0) & (y == 0))
                    FP = np.sum((y_pred == 1) & (y == 0))
                    FN = np.sum((y_pred == 0) & (y == 1))

                    TPR[i] = TP / (TP + FN)
                    FPR[i] = FP / (FP + TN)

                return np.trapezoid(TPR, FPR)
            case _:
                return (TP + TN) / y.size

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        return (x.T @ (p - y)) / y.size, np.sum(p - y) / y.size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Придатченко Павел Павлович, ПМ-34"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng or np.random.default_rng())

    # @staticmethod
    # def fit(model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int)
    # -> None:
    #     for _ in range(n_iter):
    #         grad_w, grad_b = model.grad(x, y)
    #         model.weights -= lr * grad_w
    #         model.bias -= lr * grad_b

    @staticmethod
    def fit(
        model: LinearRegression | LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_epoch: int,
        batch_size: int | None = None,
    ) -> None:

        batch = 0
        exp_num = x.shape[0]
        if batch_size is None:
            batch_size = exp_num

        for _ in range(n_epoch):
            # ind = np.random.permutation(exp_num)

            # x_shuffle = x[ind]
            # y_shuffle = y[ind]

            for batch in range(0, exp_num, batch_size):
                x_batch = x[batch : batch + batch_size]
                y_batch = y[batch : batch + batch_size]

                grad_w, grad_b = model.grad(x_batch, y_batch)
                model.weights -= lr * grad_w
                model.bias -= lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 1e-1, "batch_size": 10}
