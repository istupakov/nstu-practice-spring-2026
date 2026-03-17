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
        estimated_y = self.predict(x)
        return float(np.mean((estimated_y - y) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        estimated_y = self.predict(x)
        square_residuals = np.sum((y - estimated_y) ** 2)
        square_total_diff = np.sum((y - np.mean(y)) ** 2)
        return float(1 - square_residuals / (square_total_diff + 1e-12))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n_samples = x.shape[0]
        error_vector = self.predict(x) - y
        weight_gradient = (2 / n_samples) * (x.T @ error_vector)
        bias_gradient = np.array(2 * np.mean(error_vector))
        return weight_gradient, bias_gradient


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z_values = x @ self.weights + self.bias
        return 1 / (1 + np.exp(-z_values))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        estimated_y = self.predict(x)
        estimated_y = np.clip(estimated_y, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(estimated_y) + (1 - y) * np.log(1 - estimated_y)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        estimated_y = self.predict(x)
        class_predictions = (estimated_y >= 0.5).astype(int)
        return float(np.mean(class_predictions == y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n_samples = x.shape[0]
        error_vector = self.predict(x) - y
        weight_gradient = (1 / n_samples) * (x.T @ error_vector)
        bias_gradient = np.array(np.mean(error_vector))
        return weight_gradient, bias_gradient


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Дегтярев Кирилл Романович, ПМ-35"

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
            weight_gradient, bias_gradient = model.grad(x, y)
            model.weights -= lr * weight_gradient
            model.bias -= lr * bias_gradient
