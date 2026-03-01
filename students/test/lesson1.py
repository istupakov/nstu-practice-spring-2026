import numpy as np


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Фамилия Имя Отчество, ПМ-XX"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 1"

    @staticmethod
    def sum(x: int, y: int) -> int:
        return 2

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return A
