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
        y_prediction = self.predict(x)
        return float(np.mean((y - y_prediction) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_prediction = self.predict(x)
        P_2 = 1 - np.sum((y - y_prediction) ** 2) / np.sum((y - np.mean(y)) ** 2)
        return float(P_2)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_prediction = self.predict(x)
        MSE = y_prediction - y
        n = len(y)
        grad_weights = (2 / n) * (x.T @ MSE)
        grad_bias = (2 / n) * np.sum(MSE)
        return grad_weights, grad_bias


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
        y_prediction = self.predict(x)
        eps = np.finfo(float).eps
        y_prediction = np.clip(y_prediction, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_prediction) + (1 - y) * np.log(1 - y_prediction)))

    def metric(self, x: np.ndarray, y: np.ndarray, metric_type: str = "accuracy") -> float:
        y_prediction = self.predict(x) >= 0.5
        # Матрица ошибок
        tp = np.sum((y_prediction == 1) & (y == 1))
        fp = np.sum((y_prediction == 1) & (y == 0))
        tn = np.sum((y_prediction == 0) & (y == 0))
        fn = np.sum((y_prediction == 0) & (y == 1))

        if metric_type.lower() == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        elif metric_type.lower() == "precision":
            return tp / (tp + fp) if (tp + fp) else 0.0

        elif metric_type.lower() == "recall":
            return tp / (tp + fn) if (tp + fn) else 0.0

        elif metric_type.lower() == "f1":
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        elif metric_type.lower() == "auroc":
            y_proba = self.predict(x)

            sorted_indices = np.argsort(y_proba)
            y_sorted = y[sorted_indices]

            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)

            if n_pos == 0 or n_neg == 0:
                return 0.5

            rank_sum = 0
            for i in range(len(y_sorted)):
                if y_sorted[i] == 1:
                    rank_sum += i + 1

            auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            return float(auc)
        else:
            raise ValueError(f"Unknown metric: {metric_type}")

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_prediction = self.predict(x)
        error = y_prediction - y
        n = len(y)
        grad_weights = (1 / n) * (x.T @ error)
        grad_bias = (1 / n) * np.sum(error)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Марченко Вячеслав Иванович, ПМ-33"

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
        n_ep: int,
        batch_size: int | None = None,
    ) -> None:
        n_samples = x.shape[0]

        for _i in range(n_ep):
            if batch_size is None:
                grad_weights, grad_bias = model.grad(x, y)
                model.weights -= lr * grad_weights
                model.bias -= lr * grad_bias
            else:
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)

                    x_batch = x[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]

                    grad_weights, grad_bias = model.grad(x_batch, y_batch)
                    model.weights -= lr * grad_weights
                    model.bias -= lr * grad_bias

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.003, "batch_size": 4}
