import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))
    return tp, tn, fp, fn


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true_i = y_true.astype(int)
    pos = y_true_i == 1
    neg = y_true_i == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    scores = y_score.astype(float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    sorted_scores = scores[order]
    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean((y - pred) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float:
        pred = self.predict(x)
        return float(1 - np.mean((y - pred) ** 2) / np.var(y))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        dw = -2 * x.T @ (y - pred) / x.shape[0]
        db = -2 * np.mean(y - pred)
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
        if type is None:
            return float(np.mean(np.round(p) == y))

        y_true = y.astype(int)

        if type == "AUROC":
            return _roc_auc_score(y_true, p)

        y_pred = (p >= 0.5).astype(int)

        if type == "accuracy":
            return float(np.mean(y_pred == y_true))

        tp, tn, fp, fn = _confusion(y_true, y_pred)

        if type == "precision":
            return _safe_div(tp, tp + fp)

        if type == "recall":
            return _safe_div(tp, tp + fn)

        if type == "F1":
            prec = _safe_div(tp, tp + fp)
            rec = _safe_div(tp, tp + fn)
            return _safe_div(2 * prec * rec, prec + rec)

        return float(np.mean(y_pred == y_true))

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        dw = -x.T @ (y - pred) / x.shape[0]
        db = -np.mean(y - pred)
        return dw, np.array(db)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Мелиди Мирон Евстафьевич, ПМ-33"

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
        n = x.shape[0]
        bs = n if batch_size is None else int(batch_size)

        for _ in range(n_iter):
            for i in range(n // bs):
                x_b = x[i * bs : (i + 1) * bs]
                y_b = y[i * bs : (i + 1) * bs]
                dw, db = model.grad(x_b, y_b)
                model.weights -= lr * dw
                model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 1e-1, "batch_size": 16}
