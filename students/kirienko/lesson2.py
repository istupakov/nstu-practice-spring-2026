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
        y_pred = self.predict(x)
        return float(np.mean((y - y_pred) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        y_pred = self.predict(x)
        error = y_pred - y
        grad_w = (2 * x.T @ error) / n
        grad_b = np.array([2 * np.mean(error)])
        return grad_w, grad_b


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
        p = self.predict(x)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:

        probabilities = self.predict(x)
        predictions = (probabilities >= 0.5).astype(int)

        # Считаем матрицу ошибок
        true_positive = np.sum((predictions == 1) & (y == 1))
        false_positive = np.sum((predictions == 1) & (y == 0))
        true_negative = np.sum((predictions == 0) & (y == 0))
        false_negative = np.sum((predictions == 0) & (y == 1))

        if metric == "accuracy":
            denominator = true_positive + false_positive + true_negative + false_negative
            return (true_positive + true_negative) / denominator if denominator > 0 else 0.0

        if metric == "precision":
            denominator = true_positive + false_positive
            return true_positive / denominator if denominator > 0 else 0.0

        if metric == "recall":
            denominator = true_positive + false_negative
            return true_positive / denominator if denominator > 0 else 0.0

        if metric == "F1":
            denom_prec = true_positive + false_positive
            precision_val = true_positive / denom_prec if denom_prec > 0 else 0.0
            denom_rec = true_positive + false_negative
            recall_val = true_positive / denom_rec if denom_rec > 0 else 0.0
            denominator = precision_val + recall_val
            return 2 * precision_val * recall_val / denominator if denominator > 0 else 0.0

        if metric == "AUROC":
            positive_count = np.sum(y == 1)
            negative_count = np.sum(y == 0)

            if positive_count == 0 or negative_count == 0:
                return 0.5

            order = np.argsort(probabilities)[::-1]
            labels_sorted = y[order]

            current_tp = 0
            current_fp = 0
            previous_tpr = 0.0
            previous_fpr = 0.0
            area = 0.0

            for true_label in labels_sorted:
                if true_label == 1:
                    current_tp += 1
                else:
                    current_fp += 1

                current_tpr = current_tp / positive_count
                current_fpr = current_fp / negative_count

                width = current_fpr - previous_fpr
                height_avg = (current_tpr + previous_tpr) / 2
                area += width * height_avg

                previous_tpr = current_tpr
                previous_fpr = current_fpr

            return area

        raise ValueError(f"Unknown metric: {metric}")

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        y_pred = self.predict(x)
        error = y_pred - y
        grad_w = (x.T @ error) / n
        grad_b = np.array([np.mean(error)])
        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кириенко Илья Владимирович, ПМ-33"

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

        for _ in range(n_iter):
            if batch_size is None or batch_size >= n_samples:
                # Полный градиентный спуск
                grad_weights, grad_bias = model.grad(x, y)
                model.weights -= lr * grad_weights
                model.bias -= lr * grad_bias[0]
            else:
                num_complete_batches = n_samples // batch_size
                for i in range(num_complete_batches):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    x_batch = x[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]

                    grad_weights, grad_bias = model.grad(x_batch, y_batch)
                    model.weights -= lr * grad_weights
                    model.bias -= lr * grad_bias[0]

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.003, "batch_size": 1}
