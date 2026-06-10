import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y - self.predict(x)) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:
        # R² score
        pred = self.predict(x)
        return float(1.0 - np.mean((y - pred) ** 2) / np.var(y))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        dw = -2.0 * x.T @ (y - pred) / x.shape[0]
        db = -2.0 * np.mean(y - pred)
        return dw, np.array(db)


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(x @ self.weights + self.bias)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:
        p = self.predict(x)
        if type is None or type == "accuracy":
            return float(np.mean(np.round(p) == y))
        y_pred = (p >= 0.5).astype(int)
        if type == "precision":
            return float(precision_score(y, y_pred, zero_division=0))
        if type == "recall":
            return float(recall_score(y, y_pred))
        if type == "F1":
            return float(f1_score(y, y_pred))
        if type == "AUROC":
            return float(roc_auc_score(y, p))
        return float(accuracy_score(y, y_pred))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        dw = -x.T @ (y - pred) / x.shape[0]
        db = -np.mean(y - pred)
        return dw, np.array(db)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Батодалаев Арсалан Дабаевич, ПМ-33"

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
        bs = batch_size if batch_size is not None else n
        for _ in range(n_epoch):
            for i in range(n // bs):
                x_batch = x[i * bs : (i + 1) * bs]
                y_batch = y[i * bs : (i + 1) * bs]
                dw, db = model.grad(x_batch, y_batch)
                model.weights -= lr * dw
                model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        # Для 25 эпох, по метрике AUROC
        return {"lr": 0.005, "batch_size": 4}