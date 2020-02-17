# content of test_sample.py
from pytest import fixture


def func(x):
    return x + 1


@fixture
def fixture_1():
    return [1, 2, 3, 4, 5, 6, 7]


@fixture
def fixture_2():
    return {1: 2}


def test_answer(fixture_1):
    for items in fixture_1:
        if items == 1:
            assert func(items) == 2
        elif items == 2:
            assert func(items) == 3
        elif items == 3:
            assert func(items) == 4
        elif items == 4:
            assert func(items) == 5
        elif items == 5:
            assert func(items) == 6
        elif items == 6:
            assert func(items) == 7
        elif items == 7:
            assert func(items) == 8


def test_answer_2(fixture_2):
    for key in fixture_2:
        assert func(key) == fixture_2[key]


