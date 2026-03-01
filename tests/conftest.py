import importlib
import inspect
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import allure
import pytest

type AssignmentFinder = Callable[[], type[Assignment]]


@runtime_checkable
class Assignment(Protocol):
    @staticmethod
    def get_student() -> str: ...

    @staticmethod
    def get_topic() -> str: ...


@pytest.fixture(scope="session")
def all_assignments() -> list[type[Assignment]]:
    assignments = []
    for _, module_name, _ in pkgutil.walk_packages(["students"], "students."):
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name and issubclass(obj, Assignment):
                assignments.append(obj)
    return assignments


@pytest.fixture(
    scope="module",
    params=[
        "Воробьев Никита Александрович, ПМ-31",
        "Каяшев Валентин Константинович, ПМ-31",
        "Миллер Игорь Владиславович, ПМ-31",
        "Токмаков Дмитрий Евгеньевич, ПМ-31",
        "Урывский Александр Александрович, ПМ-31",
        "Ушатов Сергей Максимович, ПМ-31",
        "Батодалаев Арсалан Дабаевич, ПМ-32",
        "Киселев Эдуард Владиславович, ПМ-32",
        "Мелиди Мирон Евстафьевич, ПМ-32",
        "Романова Валерия Сергеевна, ПМ-32",
        "Большанин Егор Андреевич, ПМ-33",
        "Гросс Кирилл Дмитриевич, ПМ-33",
        "Кириенко Илья Владимирович, ПМ-33",
        "Колосов Константин Николаевич, ПМ-33",
        "Марченко Вячеслав Иванович, ПМ-33",
        "Наумов Дмитрий Сергеевич, ПМ-33",
        "Пантеева Валентина Ивановна, ПМ-33",
        "Разин Игорь Дмитриевич, ПМ-33",
        "Старонедов Владимир Эдуардович, ПМ-33",
        "Кузнецов Александр Павлович, ПМ-34",
        "Придатченко Павел Павлович, ПМ-34",
        "Саакян Айк Алексанович, ПМ-34",
        "Санданов Чимит Сергеевич, ПМ-34",
        "Дегтярев Кирилл Романович, ПМ-35",
        "Кудрявцев Павел Павлович, ПМ-35",
        "Кузьмин Александр Андреевич, ПМ-35",
        "Старицын Марк Вадимович, ПМ-35",
    ],
)
def student(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def assignment_finder(student: str, topic: str, all_assignments: list[type[Assignment]]) -> AssignmentFinder:
    filter_assignment = list(filter(lambda t: t.get_student() == student and t.get_topic() == topic, all_assignments))

    def get_assignment() -> type[Assignment]:
        allure.dynamic.label("topic", topic)
        allure.dynamic.label("student", student)

        assert len(filter_assignment) <= 1
        if not filter_assignment:
            pytest.skip("Assignment not found")

        path = Path(inspect.getfile(filter_assignment[0])).relative_to(Path.cwd())
        allure.attach.file(path, path.name, "text/python", path.suffix)
        allure.dynamic.link(
            f"https://github.com/istupakov/nstu-practice-spring-2026/blob/main/{path.as_posix()}",
            name=f"Source code ({path.as_posix()})",
        )

        return filter_assignment[0]

    return get_assignment
