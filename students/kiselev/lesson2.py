import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.bias + np.dot(x, self.weights)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        return np.mean((y - prediction) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        ss_res = np.sum((y - prediction) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1

        r_squared = 1 - (ss_res / ss_tot)
        return float(r_squared)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        bias_grad = -2 * np.mean(y - prediction)
        weight_grad = -2 * np.mean(x.T * (y - prediction), axis=1)
        return weight_grad, bias_grad


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-15
        prediction = np.clip(self.predict(x), eps, 1 - eps)
        return np.mean(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        predictions = self.predict(x)
        prediction_classes = predictions >= 0.5
        tp = np.sum((prediction_classes == 1) & (y == 1))
        fp = np.sum((prediction_classes == 1) & (y == 0))
        tn = np.sum((prediction_classes == 0) & (y == 0))
        fn = np.sum((prediction_classes == 0) & (y == 1))

        if type == "accuracy":
            denominator = tp + tn + fp + fn
            return (tp + tn) / denominator if denominator > 0 else 0.0

        elif type == "precision":
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        elif type == "recall":
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        elif type == "F1":
            denominator = tp + 0.5 * (fp + fn)
            return tp / denominator if denominator > 0 else 0.0

        elif type == "AUROC":
            pos_scores = predictions[y == 1]
            neg_scores = predictions[y == 0]

            n_pos = pos_scores.size
            n_neg = neg_scores.size
            if n_pos == 0 or n_neg == 0:
                return 0.5

            correct_pairs = np.sum(pos_scores[:, None] > neg_scores[None, :])
            tie_pairs = np.sum(pos_scores[:, None] == neg_scores[None, :])

            auroc = (correct_pairs + 0.5 * tie_pairs) / (n_pos * n_neg)
            return float(auroc)

        else:
            raise ValueError(f"Unknown logistic metric type: {type}")

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        bias_grad = np.mean(prediction - y)
        weight_grad = np.mean(x.T * (prediction - y), axis=1)
        return weight_grad, bias_grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Киселев Эдуард Владиславович, ПМ-33"

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
        m = x.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > m:
            batch_size = m

        for _ in range(n_epoch):
            for start in range(0, m, batch_size):
                end = start + batch_size
                x_batch = x[start:end]
                y_batch = y[start:end]

                grad_w, grad_b = model.grad(x_batch, y_batch)
                model.weights -= lr * grad_w
                model.bias -= lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.005, "batch_size": 4}
