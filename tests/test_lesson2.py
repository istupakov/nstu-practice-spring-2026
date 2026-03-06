from typing import Protocol, cast, runtime_checkable

import allure
import numpy as np
import pytest

from tests.conftest import AssignmentFinder


@runtime_checkable
class Regression(Protocol):
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def loss(self, x: np.ndarray, y: np.ndarray) -> float: ...

    def metric(self, x: np.ndarray, y: np.ndarray) -> float: ...

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class Lesson2Assignment(Protocol):
    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> Regression: ...

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> Regression: ...

    @staticmethod
    def fit(model: Regression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> None: ...


@pytest.fixture(scope="module")
def topic() -> str:
    return "Lesson 2"


@allure.label("part", "Linear")
@pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
def test_create_linear_model(assignment_finder: AssignmentFinder, num_features: int, num_points: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_linear_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0)

    assert isinstance(model, Regression)
    np.testing.assert_allclose(model.weights, weights)
    np.testing.assert_allclose(model.bias, bias)


@allure.label("part", "Linear")
@pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
def test_linear_model_predict(assignment_finder: AssignmentFinder, num_features: int, num_points: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_linear_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0)
    x = rng.random((num_points, num_features))

    expected_pred = x @ weights + bias
    np.testing.assert_allclose(model.predict(x), expected_pred)


@allure.label("part", "Linear")
@pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
def test_linear_model_loss(assignment_finder: AssignmentFinder, num_features: int, num_points: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_linear_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0)
    x = rng.random((num_points, num_features))
    y = rng.random(num_points)

    expected_loss = np.mean((y - x @ weights + bias) ** 2)
    np.testing.assert_allclose(model.loss(x, y), expected_loss)


@allure.label("part", "Linear")
@pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
def test_linear_model_metric(assignment_finder: AssignmentFinder, num_features: int, num_points: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_linear_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0)
    x = rng.random((num_points, num_features))
    y = rng.random(num_points)

    expected_metric = 1 - np.mean((y - x @ weights + bias) ** 2) / np.var(y)
    np.testing.assert_allclose(model.metric(x, y), expected_metric)


@allure.label("part", "Linear")
@pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (5, 10), (10, 100)])
def test_linear_model_grad(assignment_finder: AssignmentFinder, num_features: int, num_points: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_linear_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0)
    x = rng.random((num_points, num_features))
    y = rng.random(num_points)

    expected_pred = x @ weights + bias
    expected_dw = -2 * x.T @ (y - expected_pred) / x.shape[0]
    expected_db = -2 * np.mean(y - expected_pred)
    dw, db = model.grad(x, y)
    np.testing.assert_allclose(expected_dw, dw)
    np.testing.assert_allclose(expected_db, db)


@allure.label("part", "Linear")
@pytest.mark.parametrize(("num_features", "num_points"), [(1, 10), (3, 10)])
def test_fit_linear_model(assignment_finder: AssignmentFinder, num_features: int, num_points: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_linear_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    x = rng.random((num_points, num_features))
    y = rng.random(num_points)
    sol = np.linalg.lstsq(np.hstack((x, np.ones_like(y)[:, None])), y)[0]

    assignment.fit(model, x, y, 1e-1, 1000)
    np.testing.assert_allclose(model.weights, sol[:-1], 1e-2)
    np.testing.assert_allclose(model.bias, sol[-1], 1e-2)


@allure.label("part", "Logistic")
@pytest.mark.parametrize("num_features", [1, 10])
def test_create_logistic_model(assignment_finder: AssignmentFinder, num_features: int):
    assignment = cast(Lesson2Assignment, assignment_finder())
    model = assignment.create_logistic_model(num_features, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    weights = rng.random(num_features)
    bias = np.array(0)

    assert isinstance(model, Regression)
    np.testing.assert_allclose(model.weights, weights)
    np.testing.assert_allclose(model.bias, bias)
