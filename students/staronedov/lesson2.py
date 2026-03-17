import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.bias + x @ self.weights

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return (np.sum((y - self.predict(x)) ** 2)) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pove = np.sum((y - self.predict(x)) ** 2)
        vom = np.sum((y - np.sum(y) / y.size) ** 2)
        return 1 - (pove / vom)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        bias_grad = ((-2) * (np.sum(y - self.predict(x)))) / y.size
        weights_grad = ((-2) * (x.T) @ (y - self.predict(x))) / y.size
        return weights_grad, bias_grad


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.bias + x @ self.weights)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return -(np.sum(y * np.log(self.predict(x)) + (1 - y) * np.log(1 - self.predict(x))) / y.size)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pr = self.predict(x)
        n = 0
        iter = -1
        for i in pr:
            iter += 1
            if i >= 0.5 and y[iter] == 1 or i <= 0.5 and y[iter] == 0:
                n += 1
        return n / y.size

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        bias_grad = np.sum(self.predict(x) - y) / y.size
        weights_grad = x.T @ (self.predict(x) - y) / y.size
        return weights_grad, bias_grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старонедов Владимир Эдуардович, ПМ-33"

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
        for _ in range(n_iter):
            grad_weights, grad_bias = model.grad(x, y)

            model.weights -= lr * grad_weights
            model.bias -= lr * grad_bias
