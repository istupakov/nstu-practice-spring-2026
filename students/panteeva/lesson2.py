import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.bias + x @ self.weights

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        return float(np.mean((y - prediction) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        sum1 = np.sum((y - prediction) ** 2)
        sum2 = np.sum((y - np.mean(y)) ** 2)
        return 1 - sum1 / sum2

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        grad_bias = -2 * np.mean(y - prediction)
        grad_weights = -2 * np.mean(x.T * (y - prediction), axis=1)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.bias + x @ self.weights
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return float(np.mean(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        prediction = self.predict(x)
        bool_prediction = prediction >= 0.5
        TP = np.sum((bool_prediction == 1) & (y == 1))
        FP = np.sum((bool_prediction == 1) & (y == 0))
        FN = np.sum((bool_prediction == 0) & (y == 1))
        # TN = np.sum((bool_prediction == 0) & (y == 0))
        if type == "accuracy":
            return float(np.mean((prediction >= 0.5) == y))
        elif type == "precision":
            if TP + FP == 0:
                return 0.0
            else:
                return TP / (TP + FP)
        elif type == "recall":
            if TP + FN == 0:
                return 0.0
            else:
                return TP / (TP + FN)
        elif type == "F1":
            if TP + 0.5 * (FP + FN) == 0:
                return 0.0
            else:
                return TP / (TP + 0.5 * (FP + FN))
        else:  # type == "AUROC"
            pos_scores = prediction[y == 1]
            neg_scores = prediction[y == 0]

            total_pos = np.sum(y == 1)
            total_neg = np.sum(y == 0)

            if total_pos == 0 or total_neg == 0:
                return 0.5
            correct_pairs = np.sum(pos_scores[:, None] > neg_scores[None, :])
            tie_pairs = np.sum(pos_scores[:, None] == neg_scores[None, :])

            return (correct_pairs + 0.5 * tie_pairs) / (total_neg * total_pos)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        grad_bias = np.mean(prediction - y)
        grad_weights = np.mean(x.T * (prediction - y), axis=1)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Пантеева Валентина Ивановна, ПМ-33"

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
        if batch_size is None:
            for _ in range(n_epoch):
                grad_weights, grad_bias = model.grad(x, y)

                model.weights -= lr * grad_weights
                model.bias -= lr * grad_bias
        else:
            for _ in range(n_epoch):
                for start in range(0, x.shape[0], batch_size):
                    x_batch = x[start : start + batch_size]
                    y_batch = y[start : start + batch_size]
                    grad_weights, grad_bias = model.grad(x_batch, y_batch)

                    model.weights -= lr * grad_weights
                    model.bias -= lr * grad_bias

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.001, "batch_size": 4}
