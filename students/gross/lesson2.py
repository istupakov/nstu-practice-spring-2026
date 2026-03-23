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
        y_pred = self.predict(x)
        s1 = np.sum((y - y_pred) ** 2)
        s2 = np.sum((y - np.mean(y)) ** 2)
        return 1 - s1 / s2

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(x)
        error = y_pred - y
        grad_w = (2 * x.T @ error) / len(y)
        grad_b = 2 * np.mean(error)
        return grad_w, grad_b


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear = x @ self.weights + self.bias
        return 1 / (1 + np.exp(-linear))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:
        if type is None or type == "accuracy":
            y_pred = self.predict(x) >= 0.5
            return np.mean(y_pred == y)
        elif type == "precision":
            return self.precision(x, y)
        elif type == "recall":
            return self.recall(x, y)
        elif type == "F1":
            return self.f1(x, y)
        elif type == "AUROC":
            return self.auroc(x, y)
        else:
            raise ValueError(f"Unknown metric type: {type}")

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(x)
        error = y_pred - y
        grad_w = (x.T @ error) / len(y)
        grad_b = np.mean(error)

        return grad_w, grad_b

    # доп метрики
    def precision(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x) >= 0.5
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        return tp / (tp + fp + 1e-15)

    def recall(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x) >= 0.5
        tp = np.sum((y_pred == 1) & (y == 1))
        fn = np.sum((y_pred == 0) & (y == 1))
        return tp / (tp + fn + 1e-15)

    def f1(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.precision(x, y)
        r = self.recall(x, y)
        return 2 * p * r / (p + r + 1e-15)

    def auroc(self, x: np.ndarray, y: np.ndarray) -> float:
        y_score = self.predict(x)
        order = np.argsort(y_score)
        y = y[order]

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        rank_sum = 0
        for i in range(len(y)):
            if y[i] == 1:
                rank_sum += i + 1

        return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-15)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Гросс Кирилл Дмитриевич, ПМ-33"

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

        for _ in range(n_iter):
            if batch_size is None:
                grad_w, grad_b = model.grad(x, y)
                model.weights -= lr * grad_w
                model.bias -= lr * grad_b
            else:
                for i in range(0, n, batch_size):
                    end = min(i + batch_size, n)

                    x_batch = x[i:end]
                    y_batch = y[i:end]

                    grad_w, grad_b = model.grad(x_batch, y_batch)
                    model.weights -= lr * grad_w
                    model.bias -= lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, float]:
        return {"lr": 0.003, "batch_size": 2}
