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
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        y_pred = self.predict(x)
        error = y_pred - y
        grad_w = (2 * x.T @ error) / n
        grad_b = np.array([2 * np.mean(error)])
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
        y_pred = self.predict(x)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        y_pred_classes = (y_pred >= 0.5).astype(int)
        return float(np.mean(y == y_pred_classes))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        y_pred = self.predict(x)
        error = y_pred - y

        grad_w = (x.T @ error) / n
        grad_b = np.array([np.mean(error)])

        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кириенко Илья Владимирович, ПМ-33"

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
    def fit(model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> None:
        for _i in range(n_iter):
            grad_w, grad_b = model.grad(x, y)

            model.weights -= lr * grad_w
            model.bias -= lr * grad_b[0]
