import numpy as np


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старицын Марк Вадимович, ПМ-35"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 1"

    @staticmethod
    def sum(x: int, y: int) -> int:
        return x + y
        # return 4

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        A_invert = np.linalg.inv(A)
        return A_invert @ b
