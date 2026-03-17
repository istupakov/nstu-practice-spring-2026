import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.bias + np.dot(x, self.weights)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return np.mean((y_pred - y) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 1 - ss_res / ss_tot

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(x)
        n = len(y)
        grad_w = (2 / n) * np.dot(x.T, (y_pred - y))
        grad_b = (2 / n) * np.sum(y_pred - y)
        return grad_w, grad_b


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
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        p = self.predict(x)

        if type == "accuracy":
            y_pred = (p >= 0.5).astype(float)
            return float(np.mean(y_pred == y))

        elif type == "precision":
            y_pred = (p >= 0.5).astype(float)
            tp = np.sum((y_pred == 1) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == 0))
            if tp + fp == 0:
                return 0
            return tp / (tp + fp)

        elif type == "F1":
            y_pred = (p >= 0.5).astype(float)
            tp = np.sum((y_pred == 1) & (y == 1))
            fn = np.sum((y_pred == 0) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                return 0
            return (2 * precision * recall) / (precision + recall)

        elif type == "recall":
            y_pred = (p >= 0.5).astype(float)
            tp = np.sum((y_pred == 1) & (y == 1))
            fn = np.sum((y_pred == 0) & (y == 1))
            if tp + fn == 0:
                return 0
            return tp / (tp + fn)

        elif type == "AUROC":
            sort = np.argsort(p)[::-1]
            y_sort = y[sort]

            pos = np.sum(y == 1)
            neg = np.sum(y == 0)

            if pos == 0 or neg == 0:
                return 0.5

            tp = np.cumsum(y_sort == 1)
            fp = np.cumsum(y_sort == 0)

            tpr = tp / pos
            fpr = fp / neg

            tpr = np.concatenate(([0.0], tpr, [1.0]))
            fpr = np.concatenate(([0.0], fpr, [1.0]))

            return float(np.trapezoid(tpr, fpr))

        else:
            return 0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        n = len(y)
        grad_w = (1 / n) * np.dot(x.T, (p - y))
        grad_b = (1 / n) * np.sum(p - y)
        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Санданов Чимит Сергеевич, ПМ-34"

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

        n_samples = x.shape[0]

        if batch_size is None:
            batch_size = n_samples

        iters = int(x.shape[0] / batch_size)

        for epoch in range(n_epoch):
            epoch * 1
            for i in range(0, iters * batch_size, batch_size):
                x_b = x[i : i + batch_size]
                y_b = y[i : i + batch_size]

                gW, gB = model.grad(x_b, y_b)

                model.weights -= lr * gW
                model.bias -= lr * gB

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 1e-1, "batch_size": 5}
