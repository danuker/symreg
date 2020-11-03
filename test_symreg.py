from symreg import Regressor


def test_regressor():
    s = Regressor(10, 10)
    X = [[1, 0], [1, 1], [1, 2]]
    y = [0, 1, 2]

    s.fit(X, y)
    assert s.predict([[0, 3]]) == [3]


def test_regressor_constant():
    s = Regressor(10, 10)
    X = [[0, 0], [0, 1], [0, 2]]
    y = [1, 1, 1]

    s.fit(X, y)
    assert abs(s.predict([[0, 3]]) - [1]) < 0.1


def test_regressor_op():
    s = Regressor(50, 50)
    X = [[1, 0], [2, 1], [2, 2], [0, 2], [20, 5]]
    y = [1, 1, 0, -2, 15]   # sub $0 $1

    s.fit(X, y)
    assert s.predict([[0, 3]]) == [-3]


if __name__ == '__main__':
    # import pytest

    # pytest.main(['test_symreg.py'])
    test_regressor_op()
