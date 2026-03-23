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
        return float(-np.mean(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10)))

    def metric(self, x: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:
        probabilities = self.predict(x)

        if len(y) == 0:
            return 0.0

        predictions = (probabilities >= 0.5).astype(int)

        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        tn = np.sum((predictions == 0) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))

        total = tp + fp + tn + fn

        if metric == "accuracy":
            return float((tp + tn) / total) if total > 0 else 0.0
        elif metric == "precision":
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        elif metric == "recall":
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        elif metric == "F1":
            if (tp + fp) > 0 and (tp + fn) > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall > 0:
                    return float(2 * precision * recall / (precision + recall))
            return 0.0
        elif metric == "AUROC":
            return float(self._roc_auc_score(y, probabilities))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _roc_auc_score(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
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
        prev_tpr = 0.0
        prev_fpr = 0.0

        for i in range(n):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1

            tpr = tp / n_pos
            fpr = fp / n_neg

            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2

            prev_tpr = tpr
            prev_fpr = fpr

        return auc

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
    def fit(
        model: LinearRegression | LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_iter: int,
        batch_size: int | None = None,
    ) -> None:
        n = x.shape[0]

        for _ in range(n_iter):
            if batch_size is None:
                gradient_w, gradient_b = model.grad(x, y)
                model.weights -= lr * gradient_w
                model.bias -= lr * gradient_b
            else:
                for i in range(0, n, batch_size):
                    x_sh = x[i : min(i + batch_size, n)]
                    y_sh = y[i : min(i + batch_size, n)]

                    gradient_w, gradient_b = model.grad(x_sh, y_sh)
                    model.weights -= lr * gradient_w
                    model.bias -= lr * gradient_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, float | int]:
        return {"lr": 0.005, "batch_size": 2}
