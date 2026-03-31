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


@runtime_checkable
class Loss(Protocol):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def backward(self) -> np.ndarray: ...


@runtime_checkable
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

    @staticmethod
    def create_mse_loss() -> Loss: ...

    @staticmethod
    def create_bce_loss() -> Loss: ...

    @staticmethod
    def create_nll_loss() -> Loss: ...

    @staticmethod
    def create_cross_entropy_loss() -> Loss: ...

    @staticmethod
    def train_model(
        model: Layer, loss: Loss, x: np.ndarray, y: np.ndarray, lr: float, n_epoch: int, batch_size: int
    ) -> None: ...


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


@pytest.mark.parametrize("shape", [(1, 5), (5, 3)])
class TestLosses:
    def test_mse_loss(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = assignment_finder()
        if not isinstance(assignment, Lesson3Assignment):
            pytest.skip()
        fn = assignment.create_mse_loss()
        assert isinstance(fn, Loss)

        rng = np.random.default_rng(42)
        x = rng.random(shape, dtype=np.float32)
        y = rng.random(shape, dtype=np.float32)
        loss = np.mean((x - y) ** 2)
        dy = 2 * (x - y) / x.size

        fn_loss = fn.forward(x, y)
        np.testing.assert_allclose(fn_loss, loss, strict=True)

        fn_dy = fn.backward()
        np.testing.assert_allclose(fn_dy, dy, strict=True)

    def test_bce_loss(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = assignment_finder()
        if not isinstance(assignment, Lesson3Assignment):
            pytest.skip()
        fn = assignment.create_bce_loss()
        assert isinstance(fn, Loss)

        rng = np.random.default_rng(42)
        batch_size = shape[0]
        x = sigmoid(rng.random(shape, dtype=np.float32))
        y = rng.integers(2, size=shape)
        loss = -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))
        dy = (x - y) / (x * (1 - x)) / batch_size

        fn_loss = fn.forward(x, y)
        np.testing.assert_allclose(fn_loss, loss, strict=True)

        fn_dy = fn.backward()
        np.testing.assert_allclose(fn_dy, dy, strict=True)

    def test_nll_loss(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = assignment_finder()
        if not isinstance(assignment, Lesson3Assignment):
            pytest.skip()
        fn = assignment.create_nll_loss()
        assert isinstance(fn, Loss)

        rng = np.random.default_rng(42)
        batch_size = shape[0]
        x = log_softmax(500 - 1000 * rng.random(shape, dtype=np.float32))
        y = rng.integers(shape[-1], size=batch_size)
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1

        loss = -np.sum(x * hot_y) / batch_size
        dy = -hot_y / batch_size

        fn_loss = fn.forward(x, y)
        np.testing.assert_allclose(fn_loss, loss, strict=True)

        fn_dy = fn.backward()
        np.testing.assert_allclose(fn_dy, dy, strict=True)

    def test_cross_entropy_loss(self, assignment_finder: AssignmentFinder, shape: tuple[int]):
        assignment = assignment_finder()
        if not isinstance(assignment, Lesson3Assignment):
            pytest.skip()
        fn = assignment.create_cross_entropy_loss()
        assert isinstance(fn, Loss)

        rng = np.random.default_rng(42)
        batch_size = shape[0]
        x = 500 - 1000 * rng.random(shape, dtype=np.float32)
        y = rng.integers(shape[-1], size=batch_size)
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1

        logprobs = log_softmax(x)
        loss = -np.sum(logprobs * hot_y) / batch_size
        dy = (np.exp(logprobs) - hot_y) / batch_size

        fn_loss = fn.forward(x, y)
        np.testing.assert_allclose(fn_loss, loss, strict=True)

        fn_dy = fn.backward()
        np.testing.assert_allclose(fn_dy, dy, strict=True)


@pytest.mark.parametrize(("lr", "n_epoch", "batch_size"), [(1e-3, 1, 3), (1e-2, 1, 1), (1e-3, 2, 3)])
class TestTraining:
    def create_model(self, assignment: Lesson3Assignment) -> Layer:
        sizes = [3, 5, 4]
        rng = np.random.default_rng(42)
        return assignment.create_model(
            assignment.create_linear_layer(sizes[0], sizes[1], rng),
            assignment.create_relu_layer(),
            assignment.create_linear_layer(sizes[1], sizes[2], rng),
        )

    def train_model(
        self, model: Layer, loss: Loss, x: np.ndarray, y: np.ndarray, lr: float, n_epoch: int, batch_size: int
    ) -> None:
        idx = np.arange(batch_size, x.shape[0], batch_size)
        for _ in range(n_epoch):
            for x_batch, y_batch in zip(np.split(x, idx, axis=0), np.split(y, idx, axis=0), strict=True):
                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())

                for p, g in zip(model.parameters, model.grad, strict=True):
                    p += -lr * g

    def test_train_mse(self, assignment_finder: AssignmentFinder, lr: float, n_epoch: int, batch_size: int):
        assignment = assignment_finder()
        if not isinstance(assignment, Lesson3Assignment):
            pytest.skip()
        fn = assignment.create_mse_loss()
        model1 = self.create_model(assignment)
        model2 = self.create_model(assignment)

        rng = np.random.default_rng(42)
        x = rng.random((10, 3), dtype=np.float32)
        y = rng.random((10, 4), dtype=np.float32)

        self.train_model(model1, fn, x, y, lr, n_epoch, batch_size)
        assignment.train_model(model2, fn, x, y, lr, n_epoch, batch_size)

        for actual, expected in zip(model2.parameters, model1.parameters, strict=True):
            np.testing.assert_allclose(actual, expected, strict=True)
