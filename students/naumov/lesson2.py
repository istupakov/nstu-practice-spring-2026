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
        residuals = y - self.predict(x)
        return float(np.mean(residuals**2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        residuals = y - self.predict(x)
        total = y - np.mean(y)
        return float(1 - np.sum(residuals**2) / np.sum(total**2))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residuals = self.predict(x) - y
        n = len(y)
        grad_w = 2 * (x.T @ residuals) / n
        grad_b = np.array(2 * np.mean(residuals))
        return grad_w, grad_b


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.predict(x), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray, metric_type: str = "accuracy") -> float:
        p = self.predict(x)
        pred = (p >= 0.5).astype(int)

        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        if metric_type == "accuracy":
            total = tp + tn + fp + fn
            return float((tp + tn) / total) if total > 0 else 0.0

        elif metric_type == "precision":
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        elif metric_type == "recall":
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        elif metric_type == "F1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denominator = precision + recall
            if denominator > 0:
                return float(2 * precision * recall / denominator)
            return 0.0

        elif metric_type == "AUROC":
            pos_scores = p[y == 1]
            neg_scores = p[y == 0]
            n_pos = len(pos_scores)
            n_neg = len(neg_scores)
            if n_pos == 0 or n_neg == 0:
                return 0.5
            pos_col = pos_scores[:, np.newaxis]
            neg_row = neg_scores[np.newaxis, :]
            correct = np.sum(pos_col > neg_row)
            ties = np.sum(pos_col == neg_row)
            return float((correct + 0.5 * ties) / (n_pos * n_neg))

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residuals = self.predict(x) - y
        n = len(y)
        grad_w = (x.T @ residuals) / n
        grad_b = np.array(np.mean(residuals))
        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Наумов Дмитрий Сергеевич, ПМ-33"

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
        n_samples = x.shape[0]
        for _ in range(n_epoch):
            if batch_size is None:
                grad_w, grad_b = model.grad(x, y)
                model.weights = model.weights - lr * grad_w
                model.bias = model.bias - lr * grad_b
            else:
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    x_batch = x[start:end]
                    y_batch = y[start:end]
                    grad_w, grad_b = model.grad(x_batch, y_batch)
                    model.weights = model.weights - lr * grad_w
                    model.bias = model.bias - lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.01, "batch_size": 8}
