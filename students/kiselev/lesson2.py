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
        predction = self.predict(x)
        return np.mean((y - predction) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        ss_res = np.sum((y - prediction) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1

        r_squared = 1 - (ss_res / ss_tot)
        return float(r_squared)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        bias_grad = -2 * np.mean(y - prediction)
        weight_grad = -2 * np.mean(x.T * (y - prediction), axis=1)
        return weight_grad, bias_grad


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
        prediction = self.predict(x)
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return np.sum(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        prediction_class = (prediction >= 0.5).astype(int)
        accuracy = np.mean(prediction_class == y)
        return float(accuracy)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        bias_grad = np.sum(y - prediction)
        weight_grad = np.sum(x.T * (y - prediction), axis=1)
        return weight_grad, bias_grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Киселев Эдуард Владиславович, ПМ-33"

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
