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
        return float(np.mean((self.predict(x) - y) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (np.sum((y - self.predict(x)) ** 2) / (np.sum((y - np.mean(y)) ** 2)))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        error = prediction - y
        gradient_w = (2 * x.T @ error) / len(y)
        gradient_b = 2 * np.mean(error)
        return gradient_w, gradient_b


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -100, 100)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self.weights + self.bias)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(x)
        return -np.mean(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(x) >= 0.5
        return np.mean(predictions == y)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(x)
        error = predictions - y
        gradient_w = (x.T @ error) / len(y)
        gradient_b = np.mean(error)
        return gradient_w, gradient_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Романова Валерия Сергеевна, ПМ-34"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        if rng is None:
            rng = np.random.default_rng()
        return LinearRegression(num_features, rng)

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        if rng is None:
            rng = np.random.default_rng()
        return LogisticRegression(num_features, rng)

    @staticmethod
    def fit(model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> None:
        for _ in range(n_iter):
            gradient_w, gradient_b = model.grad(x, y)

            model.weights -= lr * gradient_w
            model.bias -= lr * gradient_b
