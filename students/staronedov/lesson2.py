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
        return (np.sum((y - self.predict(x)) ** 2)) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pove = np.sum((y - self.predict(x)) ** 2)
        vom = np.sum((y - np.sum(y) / y.size) ** 2)
        return 1 - (pove / vom)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        bias_grad = ((-2) * (np.sum(y - self.predict(x)))) / y.size
        weights_grad = ((-2) * (x.T) @ (y - self.predict(x))) / y.size
        return weights_grad, bias_grad


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.bias + x @ self.weights)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return -(np.sum(y * np.log(self.predict(x)) + (1 - y) * np.log(1 - self.predict(x))) / y.size)

    def metric(self, x: np.ndarray, y: np.ndarray, type: str = "accuracy") -> float:
        pr = self.predict(x)
        if type == "accuracy":
            n = 0
            iter = -1
            for i in pr:
                iter += 1
                if i >= 0.5 and y[iter] == 1 or i <= 0.5 and y[iter] == 0:
                    n += 1
            return n / y.size
        elif type == "precision":
            tp = 0
            fp = 0
            iter = -1
            for i in pr:
                iter += 1
                if i >= 0.5 and y[iter] == 1:
                    tp += 1
                elif i >= 0.5 and y[iter] != 1:
                    fp += 1
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif type == "recall":
            tp = 0
            fn = 0
            fp = 0
            iter = -1
            for i in pr:
                iter += 1
                if i >= 0.5 and y[iter] == 1:
                    tp += 1
                elif i <= 0.5 and y[iter] == 1:
                    fn += 1
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif type == "F1":
            tp = 0
            fn = 0
            fp = 0
            iter = -1
            for i in pr:
                iter += 1
                if i >= 0.5 and y[iter] == 1:
                    tp += 1
                elif i <= 0.5 and y[iter] == 1:
                    fn += 1
                elif i >= 0.5 and y[iter] != 1:
                    fp += 1
            return tp / (tp + (fn + fp) / 2)
        elif type == "AUROC":
            scores = pr.flatten()
            y_true = y.flatten()
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos == 0 or n_neg == 0:
                return 0.5

            order = np.argsort(scores)[::-1]
            sorted_scores = scores[order]
            sorted_y = y_true[order]

            tp = 0
            fp = 0
            points = [(0.0, 0.0)]

            i = 0
            while i < len(sorted_scores):
                current_score = sorted_scores[i]
                j = i
                pos_in_group = 0
                neg_in_group = 0
                while j < len(sorted_scores) and sorted_scores[j] == current_score:
                    if sorted_y[j] == 1:
                        pos_in_group += 1
                    else:
                        neg_in_group += 1
                    j += 1
                tp += pos_in_group
                fp += neg_in_group
                tpr = tp / n_pos
                fpr = fp / n_neg
                points.append((fpr, tpr))
                i = j
            if points[-1] != (1.0, 1.0):
                points.append((1.0, 1.0))
            auroc = 0.0
            for k in range(1, len(points)):
                fpr_prev, tpr_prev = points[k - 1]
                fpr_curr, tpr_curr = points[k]
                width = fpr_curr - fpr_prev
                avg_height = (tpr_prev + tpr_curr) / 2.0
                auroc += width * avg_height

            return auroc
        else:
            return 0.0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        bias_grad = np.sum(self.predict(x) - y) / y.size
        weights_grad = x.T @ (self.predict(x) - y) / y.size
        return weights_grad, bias_grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старонедов Владимир Эдуардович, ПМ-33"

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
        m = x.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > m:
            batch_size = m

        for _ in range(n_iter):
            for start in range(0, m, batch_size):
                end = start + batch_size
                x_batch = x[start:end]
                y_batch = y[start:end]

                grad_w, grad_b = model.grad(x_batch, y_batch)
                model.weights -= lr * grad_w
                model.bias -= lr * grad_b

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 1e-1, "batch_size": 10}
