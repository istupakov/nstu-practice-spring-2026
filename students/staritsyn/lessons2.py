import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean((y - self.predict(x)) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        up_sum = np.sum((y - self.predict(x)) ** 2)
        lower_sum = np.sum((y - np.mean(y)) ** 2)
        return 1 - up_sum / lower_sum

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        grad_weights = 2 / len(y) * np.matmul(x.T, (self.predict(x) - y))
        grad_bias = 2 / len(y) * np.sum(self.predict(x) - y)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(np.matmul(x, (self.weights)) + self.bias)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p_i = np.clip(self.predict(x), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p_i) + (1 - y) * np.log(1 - p_i))

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = "accuracy") -> float:
        p = self.predict(x)
        y_pred = (p >= 0.5).astype(int)

        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        tn = np.sum((y_pred == 0) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        if type is None or type == "accuracy":
            return (tp + tn) / len(y)

        if type == "precision":
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)

        if type == "recall":
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)

        if type == "F1":
            precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
            recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        if type == "AUROC":
            pos = np.sum(y == 1)
            neg = np.sum(y == 0)

            if pos == 0 or neg == 0:
                return 0.0

            fpr_list = []
            tpr_list = []

            for threshold in np.linspace(1.0, 0.0, 1000):
                y_thr = (p >= threshold).astype(int)

                tp_thr = np.sum((y_thr == 1) & (y == 1))
                fp_thr = np.sum((y_thr == 1) & (y == 0))
                fn_thr = np.sum((y_thr == 0) & (y == 1))
                tn_thr = np.sum((y_thr == 0) & (y == 0))

                tpr = 0.0 if (tp_thr + fn_thr) == 0 else tp_thr / (tp_thr + fn_thr)
                fpr = 0.0 if (fp_thr + tn_thr) == 0 else fp_thr / (fp_thr + tn_thr)

                tpr_list.append(tpr)
                fpr_list.append(fpr)

            return np.trapezoid(tpr_list, fpr_list)

        raise ValueError(f"Unknown metric type: {type}")

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        grad_weights = 1 / len(y) * np.matmul(x.T, (self.predict(x) - y))
        grad_bias = 1 / len(y) * np.sum(self.predict(x) - y)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старицын Марк Вадимович, ПМ-35"

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
        if batch_size is None:
            for _ in range(n_iter):
                dw, db = model.grad(x, y)
                model.weights -= lr * dw
                model.bias -= lr * db
        else:
            for _ in range(n_iter):
                for i in range(0, len(x), batch_size):
                    x_batch = x[i : i + batch_size]
                    y_batch = y[i : i + batch_size]

                    dw, db = model.grad(x_batch, y_batch)
                    model.weights -= lr * dw
                    model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.0003, "batch_size": 1}
