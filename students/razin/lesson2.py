import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        predictions = self.predict(x)

        return np.sum((predictions - y) ** 2) / n

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(x)
        sumsq_res = np.sum((y - predictions) ** 2)

        y_mean = np.sum(y) / len(y)
        sumsq_total = np.sum((y - y_mean) ** 2)

        return 1 - sumsq_res / sumsq_total

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        predictions = self.predict(x)
        error = predictions - y

        grad_weights = (2 / n) * x.T.dot(error)
        grad_bias = (2 / n) * np.sum(error)

        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x.dot(self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        predictions = self.predict(x)

        # Заменяем 0 и 1 на безопасные значения
        eps = 1e-15
        predictions = np.where(predictions == 0, eps, predictions)
        predictions = np.where(predictions == 1, 1 - eps, predictions)

        term1 = y * np.log(predictions)
        term2 = (1 - y) * np.log(1 - predictions)
        loss_value = -np.sum(term1 + term2) / n

        return loss_value

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(x)
        pred_classes = predictions >= 0.5
        correct = np.sum(pred_classes == y)
        return correct / len(y)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        predictions = self.predict(x)
        error = predictions - y

        grad_weights = (1 / n) * x.T.dot(error)
        grad_bias = (1 / n) * np.sum(error)

        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Разин Игорь Дмитриевич, ПМ-33"

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
            # Вычисляем градиенты
            grad_w, grad_b = model.grad(x, y)

            # Обновляем параметры модели
            model.weights = model.weights - lr * grad_w
            model.bias = model.bias - lr * grad_b
