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
        """R²-="""

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

    # Metric types:
    def _accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict(x) >= 0.5).astype(int) == y))

    def _threshold_convertor(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(x) >= threshold).astype(int)

    def _precision(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self._threshold_convertor(x)
        tp = np.count_nonzero((p == 1) & (y == 1))
        fp = np.count_nonzero((p == 1) & (y == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _recall(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self._threshold_convertor(x)
        tp = np.count_nonzero((p == 1) & (y == 1))
        fn = np.count_nonzero((p == 0) & (y == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _F1(self, x: np.ndarray, y: np.ndarray) -> float:
        precision = self._precision(x, y)
        recall = self._recall(x, y)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _AUROC(self, x: np.ndarray, y: np.ndarray) -> float:
        scores = self.predict(x)
        positive_scores, negative_scores = scores[y == 1], scores[y == 0]
        len_positive, len_negative = len(positive_scores), len(negative_scores)
        if len_positive == 0 or len_negative == 0:
            return 0.5
        n_correct = 0
        for p_score in positive_scores:
            n_correct += np.sum(p_score > negative_scores) + 0.5 * np.sum(p_score == negative_scores)
        return n_correct / (len_positive * len_negative)

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        match type:
            case "accuracy":
                return self._accuracy(x, y)
            case "precision":
                return self._precision(x, y)
            case "recall":
                return self._recall(x, y)
            case "F1":
                return self._F1(x, y)
            case "AUROC":
                return self._AUROC(x, y)
            case _:
                raise ValueError(f"Unknown metric type: {type}")

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
    def fit(
        model: LinearRegression | LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_epoch: int,
        batch_size: int | None = None,
    ) -> None:

        n = x.shape[0]
        batch = n if (batch_size is None or batch_size >= n) else batch_size

        for _ in range(n_epoch):
            for start in range(0, n, batch):
                end = min(start + batch, n)
                x_batch = x[start:end]
                y_batch = y[start:end]
                grad_weights, grad_bias = model.grad(x_batch, y_batch)
                model.weights -= grad_weights * lr
                model.bias -= grad_bias * lr

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.04, "batch_size": 1}
