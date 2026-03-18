import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear_part = x @ self.weights
        y_pred = linear_part + self.bias
        return y_pred

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        errors = y_pred - y
        squared_errors = errors**2
        return np.mean(squared_errors)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        y_pred = self.predict(x)
        error = y_pred - y  # знак противоположен теории, но компенсируется знаком ниже
        grad_weights = (2 / n) * (x.T @ error)
        grad_bias = (2 / n) * np.sum(error)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear_part = x @ self.weights
        z = linear_part + self.bias
        y_pred = 1 / (1 + np.exp(-z))
        return y_pred

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        p = self.predict(x)
        p = np.clip(p, 1e-15, 1 - 1e-15)  # защита от log(0)
        term = y * np.log(p)
        term1 = (1 - y) * np.log(1 - p)
        return -np.sum(term + term1) / n

    def metric(self, x: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:
        y_prob = self.predict(x)
        y_pred = (y_prob >= 0.5).astype(int)

        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        tn = np.sum((y_pred == 0) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        metrics = {
            "accuracy": (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "AUROC": self._calculate_auroc(y, y_prob),
        }

        if metric == "F1":
            p = metrics["precision"]
            r = metrics["recall"]
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if metric in metrics:
            return metrics[metric]

        raise ValueError(f"Unknown metric: {metric}")

    def _calculate_auroc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        n = len(y_true)
        if n == 0:
            return 0.5

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        sorted_indices = np.argsort(-y_scores)
        y_sorted = y_true[sorted_indices]

        auc = 0.0
        tp = 0
        fp = 0

        tpr_prev = 0.0
        fpr_prev = 0.0

        for i in range(n):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1

            tpr = tp / n_pos

            fpr = fp / n_neg

            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2

            tpr_prev = tpr
            fpr_prev = fpr
        return auc

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        y_pred = self.predict(x)
        error = y_pred - y
        grad_weights = (x.T @ error) / n
        grad_bias = np.sum(error) / n
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Колосов Константин Николаевич, ПМ-33"

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
        n_samples = x.shape[0]

        for _i in range(n_iter):
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
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.003, "batch_size": 2}
