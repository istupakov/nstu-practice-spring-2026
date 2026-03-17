import inspect
from typing import Protocol, cast, runtime_checkable

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from tests.conftest import AssignmentFinder


@runtime_checkable
class Regression(Protocol):
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def loss(self, x: np.ndarray, y: np.ndarray) -> float: ...

    def metric(self, x: np.ndarray, y: np.ndarray, type: str | None = None) -> float: ...

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class Lesson2Assignment(Protocol):
    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> Regression: ...

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> Regression: ...

    @staticmethod
    def fit(
        model: Regression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int, batch_size: int | None = None
    ) -> None: ...

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]: ...


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@pytest.fixture(scope="module")
def topic() -> str:
    return "Lesson 2"


@pytest.fixture(scope="class")
def fitted_logistic_model(batch_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    num_features = 3
    num_points = 50
    num_epoch = 100
    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0.0)
    rng = np.random.default_rng(42)
    x = rng.random((num_points, num_features))
    y = rng.integers(0, 2, num_points)
    lr = 1e-3

    if not batch_size:
        batch_size = num_points

    for _ in range(num_epoch):
        for i in range(num_points // batch_size):
            x_batch = x[i * batch_size : (i + 1) * batch_size]
            y_batch = y[i * batch_size : (i + 1) * batch_size]

            pred = sigmoid(x_batch @ weights + bias)
            dw = -x_batch.T @ (y_batch - pred) / x_batch.shape[0]
            db = -np.mean(y_batch - pred)
            weights -= lr * dw
            bias -= lr * db

    return weights, bias


class TestLinear:
    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_create_linear_model(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_linear_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(0.0)

        assert isinstance(model, Regression)
        np.testing.assert_allclose(model.weights, weights)
        np.testing.assert_allclose(model.bias, bias)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 1), (1, 10), (10, 1), (10, 21)])
    def test_linear_model_predict(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_linear_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        model.bias = bias

        expected_pred = x @ weights + bias
        if num_points == 1:
            x = x.squeeze(axis=0)
            expected_pred = expected_pred.squeeze(axis=0)

        actual_pred = model.predict(x)
        assert actual_pred.shape == expected_pred.shape
        np.testing.assert_allclose(actual_pred, expected_pred)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_linear_model_loss(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_linear_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        y = rng.random(num_points)
        model.bias = bias

        expected_loss = np.mean((y - (x @ weights + bias)) ** 2)
        np.testing.assert_allclose(model.loss(x, y), expected_loss)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_linear_model_metric(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_linear_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        y = rng.random(num_points)
        model.bias = bias

        expected_metric = 1 - np.mean((y - (x @ weights + bias)) ** 2) / np.var(y)
        np.testing.assert_allclose(model.metric(x, y), expected_metric)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_linear_model_grad(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_linear_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        y = rng.random(num_points)
        model.bias = bias

        expected_pred = x @ weights + bias
        expected_dw = -2 * x.T @ (y - expected_pred) / x.shape[0]
        expected_db = -2 * np.mean(y - expected_pred)
        dw, db = model.grad(x, y)
        np.testing.assert_allclose(expected_dw, dw)
        np.testing.assert_allclose(expected_db, db)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (3, 10)])
    def test_fit_linear_model(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_linear_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        x = rng.random((num_points, num_features))
        y = rng.random(num_points)
        sol = np.linalg.lstsq(np.hstack((x, np.ones_like(y)[:, None])), y)[0]

        assignment.fit(model, x, y, 1e-1, 1000)
        np.testing.assert_allclose(model.weights, sol[:-1], 1e-2)
        np.testing.assert_allclose(model.bias, sol[-1], 1e-2)


class TestLogistic:
    @pytest.mark.parametrize("num_features", [1, 10])
    def test_create_logistic_model(self, assignment_finder: AssignmentFinder, num_features: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(0.0)

        assert isinstance(model, Regression)
        np.testing.assert_allclose(model.weights, weights)
        np.testing.assert_allclose(model.bias, bias)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 1), (1, 10), (10, 1), (10, 21)])
    def test_logistic_model_predict(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        model.bias = bias

        expected_pred = sigmoid(x @ weights + bias)
        if num_points == 1:
            x = x.squeeze(axis=0)
            expected_pred = expected_pred.squeeze(axis=0)

        np.testing.assert_allclose(model.predict(x), expected_pred)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_logistic_model_loss(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)
        model.bias = bias

        p = sigmoid(x @ weights + bias)
        expected_loss = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        np.testing.assert_allclose(model.loss(x, y), expected_loss)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_logistic_model_metric(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)
        model.bias = bias

        expected_metric = np.mean(np.round(sigmoid(x @ weights + bias)) == y)
        np.testing.assert_allclose(model.metric(x, y), expected_metric)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_logistic_model_grad(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        weights = rng.random(num_features)
        bias = np.array(rng.random())
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)
        model.bias = bias

        expected_pred = sigmoid(x @ weights + bias)
        expected_dw = -x.T @ (y - expected_pred) / x.shape[0]
        expected_db = -np.mean(y - expected_pred)
        dw, db = model.grad(x, y)
        np.testing.assert_allclose(expected_dw, dw)
        np.testing.assert_allclose(expected_db, db)

    @pytest.fixture(scope="class")
    def batch_size(self) -> None:
        return None

    def test_fit_logistic_model(
        self,
        assignment_finder: AssignmentFinder,
        batch_size: int | None,
        fitted_logistic_model: tuple[np.ndarray, np.ndarray],
    ):
        assignment = cast(Lesson2Assignment, assignment_finder())
        num_features = 3
        num_points = 50
        num_epoch = 100

        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)

        assignment.fit(model, x, y, 1e-3, num_epoch)
        np.testing.assert_allclose(model.weights, fitted_logistic_model[0])
        np.testing.assert_allclose(model.bias, fitted_logistic_model[1])


class TestLogisticPart2:
    @pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "F1"])
    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_simple_metrics(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int, metric: str):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))
        sig = inspect.signature(model.metric)
        if len(sig.parameters) < 3:
            pytest.skip()

        rng = np.random.default_rng(42)
        weights = rng.normal(0, 1, num_features)
        bias = np.array(rng.normal(0, 1))
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)
        model.weights = weights
        model.bias = bias

        y_pred = sigmoid(x @ weights + bias) >= 0.5
        if metric == "accuracy":
            expected_metric = accuracy_score(y, y_pred)
        if metric == "precision":
            expected_metric = precision_score(y, y_pred)
        if metric == "recall":
            expected_metric = recall_score(y, y_pred)
        if metric == "F1":
            expected_metric = f1_score(y, y_pred)
        np.testing.assert_allclose(model.metric(x, y, metric), expected_metric)

    @pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
    def test_roc_auc_metric(self, assignment_finder: AssignmentFinder, num_features: int, num_points: int):
        assignment = cast(Lesson2Assignment, assignment_finder())
        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))
        sig = inspect.signature(model.metric)
        if len(sig.parameters) < 3:
            pytest.skip()

        rng = np.random.default_rng(42)
        weights = rng.normal(0, 1, num_features)
        bias = np.array(rng.normal(0, 1))
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)
        model.weights = weights
        model.bias = bias

        y_pred = sigmoid(x @ weights + bias)
        expected_metric = roc_auc_score(y, y_pred)
        np.testing.assert_allclose(model.metric(x, y, "AUROC"), expected_metric)

    @pytest.fixture(scope="class", params=[1, 5, 10, None])
    def batch_size(self, request: pytest.FixtureRequest) -> int | None:
        return request.param

    def test_fit_logistic_model(
        self,
        assignment_finder: AssignmentFinder,
        batch_size: int | None,
        fitted_logistic_model: tuple[np.ndarray, np.ndarray],
    ):
        assignment = cast(Lesson2Assignment, assignment_finder())
        sig = inspect.signature(assignment.fit)
        if len(sig.parameters) < 6:
            pytest.skip()

        num_features = 3
        num_points = 50
        num_epoch = 100

        model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        x = rng.random((num_points, num_features))
        y = rng.integers(0, 2, num_points)

        assignment.fit(model, x, y, 1e-3, num_epoch, batch_size)
        np.testing.assert_allclose(model.weights, fitted_logistic_model[0])
        np.testing.assert_allclose(model.bias, fitted_logistic_model[1])

    def test_iris_hyperparameters(self, assignment_finder: AssignmentFinder):
        assignment = cast(Lesson2Assignment, assignment_finder())
        if not getattr(assignment, "get_iris_hyperparameters", None):
            pytest.skip()

        hp = assignment.get_iris_hyperparameters()
        assert isinstance(hp.get("lr"), float)
        assert isinstance(hp.get("batch_size"), int)
