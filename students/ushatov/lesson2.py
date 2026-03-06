import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p_error = y - self.predict(x)
        return float(np.mean(np.dot(p_error, p_error) / len(y)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        """R²"""

        p_error = y - self.predict(x)
        square_p_error = np.dot(p_error, p_error)

        total = y - np.mean(y)
        square_total = np.dot(total, total)

        return float(1 - square_p_error / square_total) if square_total != 0 else 0.0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        grad_error = self.predict(x) - y
        grad_coefficient = 2.0 / n
        grad_weights = grad_coefficient * np.dot(x.T, grad_error)
        grad_bias = grad_coefficient * np.sum(grad_error)

        return grad_weights, np.array(grad_bias)


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        (P(y=1|X)).
        sigmoid(z)= β₀ + β₁x₁ + ... + βkxk. np.dot(x, self.weights) + self.bias
        P = 1 / (1 + exp(-z))
        """
        return 1 / (1 + np.exp(-(np.dot(x, self.weights) + self.bias)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        log_loss = (np.dot(y, np.log(p)) + np.dot((1 - y), np.log(1 - p))) / len(y)

        return float(-log_loss)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        predictions = (self.predict(x) >= 0.5).astype(int)
        tp = np.sum((predictions == 1) & (y == 1)) # TruePositive
        tn = np.sum((predictions == 0) & (y == 0)) # TrueNegative
        fp = np.sum((predictions == 1) & (y == 0)) # FalsePositive
        fn = np.sum((predictions == 0) & (y == 1)) # FalseNegative

        total = tp + tn + fp + fn
        if total == 0:
            return 0.0
        accuracy = (tp + tn) / total
        return float(accuracy)
        """
        return float(np.mean((self.predict(x) >= 0.5).astype(int) == y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        grad_error = self.predict(x) - y
        grad_coefficient = 1 / n
        grad_weights = grad_coefficient * np.dot(x.T, grad_error)
        grad_bias = grad_coefficient * np.sum(grad_error)

        return grad_weights, np.array(grad_bias)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Ушатов Сергей Максимович, ПМ-31"

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
            model.weights -= grad_weights * lr
            model.bias -= grad_bias * lr
