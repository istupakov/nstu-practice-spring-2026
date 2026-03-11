import numpy as np


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Дегтярев Кирилл Романович, ПМ-35"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 1"

    @staticmethod
    def sum(x: int, y: int) -> int:
        return x + y

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(A, b)
