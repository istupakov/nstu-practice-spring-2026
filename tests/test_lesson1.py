from typing import Protocol, runtime_checkable

import allure
import numpy as np
import pytest

from tests.conftest import AssignmentFinder


@runtime_checkable
class Lesson1Assignment(Protocol):
    @staticmethod
    def sum(x: int, y: int) -> int: ...

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray: ...


@pytest.fixture(scope="module")
def topic() -> str:
    return "Lesson 1"


@allure.label("part", "sum")
def test_sum(assignment_finder: AssignmentFinder):
    assignment = assignment_finder()
    assert isinstance(assignment, Lesson1Assignment)
    assert assignment.sum(2, 2) == 4


@allure.label("part", "solve")
@pytest.mark.parametrize("n", [1, 2, 10])
def test_solve(assignment_finder: AssignmentFinder, n: int):
    assignment = assignment_finder()
    assert isinstance(assignment, Lesson1Assignment)
    rng = np.random.default_rng(0)
    A = rng.random((n, n), dtype=np.float32)
    x = rng.random(n, dtype=np.float32)
    b = A @ x
    np.testing.assert_allclose(assignment.solve(A, b), x, atol=1e-6)
