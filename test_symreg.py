import random
import pandas as pd

from symreg import Regressor


def test_regressor():
    r = Regressor(duration=0.1)
    X = [[1, 0], [1, 1], [1, 2]]
    y = [0, 1, 2]

    r.fit(X, y)
    assert r.predict([[0, 3]]) == [3]


def test_regressor_constant():
    r = Regressor(duration=0.1)
    X = [[0, 0], [0, 1], [0, 2]]
    y = [1, 1, 1]

    r.fit(X, y)

    assert r.predict([[0, 3]]) == [1]


def test_regressor_constant_pandas():
    r = Regressor(duration=0.1)
    X = pd.DataFrame([[0, 0], [0, 1], [0, 2]])
    y = pd.Series([1, 1, 1])

    r.fit(X, y)

    assert r.predict([[0, 3]]) == [1]


def test_regressor_op():
    r = Regressor(duration=.3)
    X = [(random.gauss(0, 10), random.gauss(0, 10)) for _ in range(5)]
    y = [a-b for a, b in X]  # Ideal program should be: 'sub $0 $1'

    r.fit(X, y)
    assert r.predict([[0, 3]]) == [-3], \
        "These `r.fit` tests may fail, but if every test passes at least ONCE, then the code is fine.\n" \
        "If they give you trouble, increase the duration.\n" \
        "Hopefully we'll figure out how deterministic seeding works."


def test_args_are_passed():
    args = {
        'n': 14,
        'zero_program_chance': 0.01321,
        'grow_root_mutation_chance': .02567,
        'grow_leaf_mutation_chance': .02578,
        'int_std': 4,
        'float_std': 1,
    }

    r = Regressor(
        duration=.1,
        verbose=False,
        **args
    )

    r.fit([[1]], [1])
    ga = r._ga
    progs = list(r.results())
    prog = progs[0]['program']
    assert ga.n == args['n']
    assert ga.zero_program_chance == args['zero_program_chance']
    assert prog.grow_root_mutation_chance == args['grow_root_mutation_chance']
    assert prog.grow_leaf_mutation_chance == args['grow_leaf_mutation_chance']
    assert prog.int_std == args['int_std']
    assert prog.float_std == args['float_std']


if __name__ == '__main__':
    import pytest

    pytest.main(['test_symreg.py', '--color=yes'])
