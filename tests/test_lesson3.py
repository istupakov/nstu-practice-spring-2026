from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

import numpy as np
import pytest

from tests.conftest import AssignmentFinder


@runtime_checkable
class Layer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def backward(self, dy: np.ndarray) -> np.ndarray: ...

    @property
    def parameters(self) -> Sequence[np.ndarray]: ...

    @property
    def grad(self) -> Sequence[np.ndarray]: ...


class Lesson3Assignment(Protocol):
    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> Layer: ...

    @staticmethod
    def create_relu_layer() -> Layer: ...

    @staticmethod
    def create_sigmoid_layer() -> Layer: ...

    @staticmethod
    def create_logsoftmax_layer() -> Layer: ...

    @staticmethod
    def create_model(*layers: Layer) -> Layer: ...


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


@pytest.fixture(scope="module")
def topic() -> str:
    return "Lesson 3"


@pytest.mark.parametrize(("in_features", "out_features"), [(1, 1), (1, 3), (3, 1), (2, 5)])
class TestLinearLayer:
    def test_create(self, assignment_finder: AssignmentFinder, in_features: int, out_features: int):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_linear_layer(in_features, out_features, np.random.default_rng(42))

        assert isinstance(model, Layer)
        rng = np.random.default_rng(42)
        k = np.sqrt(1 / in_features)
        weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        bias = rng.uniform(-k, k, out_features).astype(np.float32)

        model_weights, model_bias = model.parameters
        np.testing.assert_allclose(model_weights, weights, strict=True)
        np.testing.assert_allclose(model_bias, bias, strict=True)

    @pytest.mark.parametrize("batch_size", [None, 1, 5])
    def test_forward(self, assignment_finder: AssignmentFinder, in_features: int, out_features: int, batch_size: int):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_linear_layer(in_features, out_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        k = np.sqrt(1 / in_features)
        weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        bias = rng.uniform(-k, k, out_features).astype(np.float32)
        x = rng.random((batch_size or 1, in_features), dtype=np.float32)
        if batch_size is None:
            x = x.squeeze(axis=0)
        y = x @ weights.T + bias

        model_y = model.forward(x)
        np.testing.assert_allclose(model_y, y, strict=True)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_backward(self, assignment_finder: AssignmentFinder, in_features: int, out_features: int, batch_size: int):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_linear_layer(in_features, out_features, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        k = np.sqrt(1 / in_features)
        weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        x = rng.random((batch_size or 1, in_features), dtype=np.float32)
        dy = rng.random((batch_size, out_features), dtype=np.float32)

        dw = dy.T @ x
        db = np.sum(dy, axis=0)
        dx = dy @ weights

        model.forward(x)
        model_dx = model.backward(dy)
        model_dw, model_db = model.grad
        np.testing.assert_allclose(model_dx, dx, strict=True)
        np.testing.assert_allclose(model_dw, dw, strict=True)
        np.testing.assert_allclose(model_db, db, strict=True)


class TestReLULayer:
    def test_create(self, assignment_finder: AssignmentFinder):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_relu_layer()

        assert isinstance(model, Layer)
        assert model.parameters == ()

    @pytest.mark.parametrize("shape", [(1,), (1, 5), (5, 3)])
    def test_forward(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_relu_layer()

        rng = np.random.default_rng(42)
        x = 5 - 10 * rng.random(shape, dtype=np.float32)
        y = np.maximum(x, 0)

        model_y = model.forward(x)
        np.testing.assert_allclose(model_y, y, strict=True)

    @pytest.mark.parametrize("shape", [(1,), (1, 5), (5, 3)])
    def test_backward(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_relu_layer()

        rng = np.random.default_rng(42)
        x = 5 - 10 * rng.random(shape, dtype=np.float32)
        y = np.maximum(x, 0)
        dy = rng.random(shape, dtype=np.float32)
        dx = dy * np.sign(y)

        model.forward(x)
        model_dx = model.backward(dy)
        np.testing.assert_allclose(model_dx, dx, strict=True)
        assert model.grad == ()


class TestSigmoidLayer:
    def test_create(self, assignment_finder: AssignmentFinder):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_sigmoid_layer()

        assert isinstance(model, Layer)
        assert model.parameters == ()

    @pytest.mark.parametrize("shape", [(1,), (1, 5), (5, 3)])
    def test_forward(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_sigmoid_layer()

        rng = np.random.default_rng(42)
        x = 5 - 10 * rng.random(shape, dtype=np.float32)
        y = sigmoid(x)

        model_y = model.forward(x)
        np.testing.assert_allclose(model_y, y, strict=True)

    @pytest.mark.parametrize("shape", [(1,), (1, 5), (5, 3)])
    def test_backward(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_sigmoid_layer()

        rng = np.random.default_rng(42)
        x = 5 - 10 * rng.random(shape, dtype=np.float32)
        y = sigmoid(x)
        dy = rng.random(shape, dtype=np.float32)
        dx = dy * y * (1 - y)

        model.forward(x)
        model_dx = model.backward(dy)
        np.testing.assert_allclose(model_dx, dx, strict=True)
        assert model.grad == ()


class TestLogSoftmaxLayer:
    def test_create(self, assignment_finder: AssignmentFinder):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_logsoftmax_layer()

        assert isinstance(model, Layer)
        assert model.parameters == ()

    @pytest.mark.parametrize("shape", [(2,), (1, 5), (5, 3)])
    def test_forward(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_logsoftmax_layer()

        rng = np.random.default_rng(42)
        x = 500 - 1000 * rng.random(shape, dtype=np.float32)
        y = log_softmax(x)

        model_y = model.forward(x)
        np.testing.assert_allclose(model_y, y, strict=True)

    @pytest.mark.parametrize("shape", [(2,), (1, 5), (5, 3)])
    def test_backward(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = cast(Lesson3Assignment, assignment_finder())
        model = assignment.create_logsoftmax_layer()

        rng = np.random.default_rng(42)
        x = 500 - 1000 * rng.random(shape, dtype=np.float32)
        y = log_softmax(x)
        dy = rng.random(shape, dtype=np.float32)
        dx = dy - (np.exp(y) * np.sum(dy, axis=-1, keepdims=True))

        model.forward(x)
        model_dx = model.backward(dy)
        np.testing.assert_allclose(model_dx, dx, strict=True)
        assert model.grad == ()


class TestModel:
    def create_layers(self, assignment: Lesson3Assignment) -> list[Layer]:
        sizes = [2, 3, 4, 2]
        rng = np.random.default_rng(42)
        return [
            assignment.create_linear_layer(sizes[0], sizes[1], rng),
            assignment.create_relu_layer(),
            assignment.create_linear_layer(sizes[1], sizes[2], rng),
            assignment.create_sigmoid_layer(),
            assignment.create_linear_layer(sizes[2], sizes[3], rng),
            assignment.create_logsoftmax_layer(),
        ]

    def test_create(self, assignment_finder: AssignmentFinder):
        assignment = cast(Lesson3Assignment, assignment_finder())

        layers = self.create_layers(assignment)
        model = assignment.create_model(*self.create_layers(assignment))
        parameters = [p for layer in layers for p in layer.parameters]

        assert isinstance(model, Layer)
        model_parameters = model.parameters
        for actual, expected in zip(model_parameters, parameters, strict=True):
            np.testing.assert_allclose(actual, expected, strict=True)

    @pytest.mark.parametrize("batch_size", [None, 1, 5])
    def test_forward(self, assignment_finder: AssignmentFinder, batch_size: int):
        assignment = cast(Lesson3Assignment, assignment_finder())

        layers = self.create_layers(assignment)
        model = assignment.create_model(*self.create_layers(assignment))

        rng = np.random.default_rng(42)
        x = rng.random((batch_size or 1, 2), dtype=np.float32)
        if batch_size is None:
            x = x.squeeze(axis=0)
        y = x
        for layer in layers:
            y = layer.forward(y)

        model_y = model.forward(x)
        np.testing.assert_allclose(model_y, y, strict=True)

    @pytest.mark.parametrize("batch_size", [1, 5])
    def test_backward(self, assignment_finder: AssignmentFinder, batch_size: int):
        assignment = cast(Lesson3Assignment, assignment_finder())

        layers = self.create_layers(assignment)
        model = assignment.create_model(*self.create_layers(assignment))

        rng = np.random.default_rng(42)
        x = rng.random((batch_size, 2), dtype=np.float32)
        dy = rng.random((batch_size, 2), dtype=np.float32)
        y = x
        for layer in layers:
            y = layer.forward(y)
        dx = dy
        for layer in layers[::-1]:
            dx = layer.backward(dx)
        grad = [g for layer in layers for g in layer.grad]

        model.forward(x)
        model_dx = model.backward(dy)
        np.testing.assert_allclose(model_dx, dx, strict=True)

        model_grad = model.grad
        for actual, expected in zip(model_grad, grad, strict=True):
            np.testing.assert_allclose(actual, expected, strict=True)
