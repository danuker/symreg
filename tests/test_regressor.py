import random
from time import time

import pandas as pd

from symreg.regressor import Regressor


def make_seeded_regressor(seed=0, **args):
    random.seed(seed)
    return Regressor(**args)


def test_regressor():
    r = make_seeded_regressor(generations=3)
    X = [[1, 0], [1, 1], [1, 2]]
    y = [0, 1, 2]

    r.fit(X, y)
    assert r.predict([[0, 3]]) == [3]


def test_regressor_constant():
    r = make_seeded_regressor(generations=2)
    X = [[0, 0], [0, 1], [0, 2]]
    y = [1, 1, 1]

    r.fit(X, y)

    assert r.predict([[0, 3]]) == [1]


def test_regressor_constant_pandas():
    r = make_seeded_regressor(generations=2)
    X = pd.DataFrame([[0, 0], [0, 1], [0, 2]])
    y = pd.Series([1, 1, 1])

    r.fit(X, y)

    assert r.predict([[0, 3]]) == [1]


def test_regressor_op():
    r = make_seeded_regressor(generations=10)

    X = [(random.gauss(0, 10), random.gauss(0, 10)) for _ in range(5)]
    y = [a - b for a, b in X]  # Ideal program should be: 'sub $0 $1'
    r.fit(X, y)
    assert r.predict([[1, 3]]) == [-2], \
        "If it fails, try another seed or more generations"


def test_deterministic():
    """
    Do not use sets, only OrderedSets
    Fortunately, dicts iterate by order of adding data.
    https://stackoverflow.com/questions/36317520/seeded-python-rng-showing-non-deterministic-behavior-with-sets
    """

    X = [(random.gauss(0, 10), random.gauss(0, 10)) for _ in range(5)]
    y = [a*7.11 - b ** 7.13212 for a, b in X]  # Ideal program should be: 'sub $0 $1'

    r = make_seeded_regressor(generations=10, n=10)
    r.fit(X, y)

    g = make_seeded_regressor(generations=10, n=10)
    g.fit(X, y)

    assert r.results() == g.results()


def test_args_are_passed():
    args = {
        'duration': .01,
        'verbose': False,
        'n': 14,
        'zero_program_chance': .11,
        'hoist_mutation_chance': .12,
        'grow_root_mutation_chance': .13,
        'grow_leaf_mutation_chance': .14,
        'complete_tree_as_new_subtree_chance': .5,
        'mutation_children': .6,
        'crossover_children': .7,
        'simplify_chance': .8,
        'int_std': 4,
        'float_std': 1,
    }

    r = make_seeded_regressor(**args)
    r.fit([[1]], [1])

    ga = r._ga
    progs = list(r.results())
    prog = progs[0]['program']
    assert ga.conf.n == args['n']
    assert ga.conf.zero_program_chance == args['zero_program_chance']
    assert ga.conf.mutation_children == args['mutation_children']
    assert ga.conf.crossover_children == args['crossover_children']
    assert ga.conf.simplify_chance == args['simplify_chance']

    assert prog.conf.hoist_mutation_chance == args['hoist_mutation_chance']
    assert prog.conf.grow_root_mutation_chance == args['grow_root_mutation_chance']
    assert prog.conf.grow_leaf_mutation_chance == args['grow_leaf_mutation_chance']
    assert prog.conf.complete_tree_as_new_subtree_chance == args['complete_tree_as_new_subtree_chance']
    assert prog.conf.int_std == args['int_std']
    assert prog.conf.float_std == args['float_std']


def test_pandas_columns_as_arg_names():
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [0, 1, 0, 1, 0]})
    y = X['a'] - X['b']

    r = make_seeded_regressor(generations=3, n=10)
    r.fit(X, y)
    program = str(r.results()[-1]['program'])
    assert '$a' in program or '$b' in program
    assert ('$0' not in program) and ('$1' not in program)


def test_pandas_series_input():
    X = pd.Series([1, 2, 3, 4, 5])
    y = X * 2

    r = make_seeded_regressor(generations=10)
    r.fit(X, y)
    assert all(r.predict(X) == y)


def test_stopping_conditions():
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [0, 1, 0, 1, 0]})
    y = X['a'] - X['b']

    r = make_seeded_regressor(generations=3, n=2)
    r.fit(X, y)
    assert r.training_details['generations'] == 3

    target_duration = .1
    r = make_seeded_regressor(duration=target_duration, n=2)
    start = time()
    r.fit(X, y)
    duration = time() - start
    assert (duration - target_duration) < 0.02

    target_stagnation = 2
    r = make_seeded_regressor(stagnation_limit=target_stagnation, n=2)
    r.fit(X, y)
    assert r.training_details['stagnated_generations'] == target_stagnation


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
    # test_stopping_conditions()
