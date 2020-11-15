import random
from time import time

import pandas as pd

from symreg.regressor import Regressor


def test_regressor():
    random.seed(0)
    r = Regressor(generations=2)
    X = [[1, 0], [1, 1], [1, 2]]
    y = [0, 1, 2]

    r.fit(X, y)
    assert r.predict([[0, 3]]) == [3]


def test_regressor_constant():
    random.seed(0)
    r = Regressor(generations=2)
    X = [[0, 0], [0, 1], [0, 2]]
    y = [1, 1, 1]

    r.fit(X, y)

    assert r.predict([[0, 3]]) == [1]


def test_regressor_constant_pandas():
    random.seed(0)
    r = Regressor(generations=2)
    X = pd.DataFrame([[0, 0], [0, 1], [0, 2]])
    y = pd.Series([1, 1, 1])

    r.fit(X, y)

    assert r.predict([[0, 3]]) == [1]


def test_regressor_op():
    random.seed(2)
    X = [(random.gauss(0, 10), random.gauss(0, 10)) for _ in range(5)]
    y = [a - b for a, b in X]  # Ideal program should be: 'sub $0 $1'

    r = Regressor(generations=3, verbose=True)
    r.fit(X, y)
    assert r.predict([[1, 3]]) == [-2], \
        "If it fails, try another seed or more generations"


def test_deterministic():
    """
    Do not use sets, only OrderedSets
    Do not use lists, only tuples
    https://stackoverflow.com/questions/36317520/seeded-python-rng-showing-non-deterministic-behavior-with-sets
    After modifying code, it is OK to update the prediction but only once.
    """

    random.seed(0)
    X = [(random.gauss(0, 10), random.gauss(0, 10)) for _ in range(5)]
    y = [a + b for a, b in X]  # Ideal program should be: 'sub $0 $1'

    r = Regressor(generations=10, n=4, verbose=True)
    r.fit(X, y)
    assert r.predict([[0, 3]]) == [-6.515598876122974]


def test_args_are_passed():
    random.seed(0)
    args = {
        'duration': .1,
        'verbose': False,
        'n': 14,
        'zero_program_chance': 0.01321,
        'grow_root_mutation_chance': .02567,
        'grow_leaf_mutation_chance': .02578,
        'int_std': 4,
        'float_std': 1,
    }

    r = Regressor(**args)

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


def test_pandas_columns_as_arg_names():
    random.seed(0)
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [0, 1, 0, 1, 0]})
    y = X['a'] - X['b']

    r = Regressor(generations=3)
    r.fit(X, y)
    program = str(r.results()[-1]['program'])
    assert '$a' in program or '$b' in program
    assert ('$0' not in program) and ('$1' not in program)


def test_stopping_conditions():
    random.seed(0)
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [0, 1, 0, 1, 0]})
    y = X['a'] - X['b']

    r = Regressor(generations=3)
    r.fit(X, y)
    assert r.training_details['generations'] == 3

    target_duration = .2
    r = Regressor(duration=target_duration)
    start = time()
    r.fit(X, y)
    duration = time() - start
    assert (duration - target_duration) < 0.1


def test_not_using_seed_everywhere_but_pytest_fixture():
    assert False

def test_ga_steps_always_one_remove():
    assert False


if __name__ == '__main__':
    import pytest

    pytest.main(['tests/test_regressor.py', '--color=yes'])
    # test_regressor()
    # test_regressor_constant()
    # test_regressor_constant_pandas()
    # test_regressor_op()
    # test_args_are_passed()
    # test_pandas_columns_as_arg_names()
    # test_deterministic()
