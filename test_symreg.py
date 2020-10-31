from symreg import Regressor


def test_regressor():
    s = Regressor()
    X = [[0, 0], [0, 1], [0, 2]]
    y = [0, 1, 2]

    s.fit(X, y)
    assert s.predict([0, 3]) == 3


if __name__ == '__main__':
    import pytest

    pytest.main(['test_symreg.py'])
