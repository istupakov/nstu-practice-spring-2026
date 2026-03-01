import numpy as np


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Разин Игорь Дмитриевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 1"

    @staticmethod
    def sum(x: int, y: int) -> int:
        return x + y

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(b)
        Ab = np.column_stack([A, b])

        # Прямой ход с выбором главного элемента
        for i in range(n):
            # Ищем максимальный элемент в столбце i
            max_row = np.argmax(np.abs(Ab[i:, i])) + i

            # Меняем строки через временную переменную
            if max_row != i:
                temp = Ab[i].copy()
                Ab[i] = Ab[max_row].copy()
                Ab[max_row] = temp

            # Делим i-ю строку на диагональный элемент
            Ab[i] = Ab[i] / Ab[i][i]

            # Вычитаем i-ю строку из всех нижних строк
            for k in range(i + 1, n):
                Ab[k] = Ab[k] - Ab[k][i] * Ab[i]

        # Обратный ход
        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = Ab[i][n] - np.sum(Ab[i][i + 1 : n] * x[i + 1 :])

        return x
