import random

from symreg.ga import Program, fitness
from symreg.regressor import Configuration
import numpy as np
import pandas as pd
from pytest import raises


def test_program():
    assert Program('3')._p == 3
    assert Program('$0', 1)._p == '$0'
    assert Program('add 3 4')._p == (np.add, 3, 4)
    assert Program('add div 3 2 4')._p == (np.add, (np.divide, 3, 2), 4)
    assert Program('3').eval() == [3]
    assert Program('add 3 4').eval() == [7]
    assert Program('sub 5 4').eval() == [1]
    assert Program('add add 1 2 3').eval() == [6]
    assert Program('add 1 $0', 1).eval([2]) == [3]
    assert Program('sub $0 neg -1', 1).eval([[1]]) == [0]
    assert Program('sub neg -1 $0', 1).eval([[-1]]) == [2]
    assert Program('add 3 sub 5 4').eval() == [4]
    assert Program('exp 1').eval() == [2.718281828459045]
    assert Program('div $0 $1', 2).eval([-1, 2]) == [float(-0.5)]
    assert Program('div $0 4', 1).eval([1]) == [0.25]
    assert Program('div $0 0', 1).eval([1]) == [float('inf')]
    assert Program('div $0 0', 1).eval([-1]) == [float('-inf')]
    assert Program('div $0 -inf', 1).eval([1]) == [0.0]
    assert {Program('123'), Program('123')} == {Program('123')}
    assert (Program('add $0 1', 1).eval(np.array([[0, 1]])) == [1, 2]).all()
    assert (Program('add $0 $1', 2).eval(np.array([[0, 2], [1, 3]])) == np.array([1, 5])).all()
    assert (Program('add $0 $1', 2).eval([[0, 1, 4], [2, 3, 5]])
            == [2, 4, 9]).all()
    assert (Program('add $0 1', 1).eval([[0, 1]]) == [1, 2]).all()

    assert (Program('$1', 2).eval([[0, 0, 0], [0, 1, 2]]) == pd.Series([0, 1, 2])).all()

    # Program too long should be a problem
    with raises(ValueError):
        Program('add 1 2 3', 1).eval([1])

    # More args than needed should be no problem
    assert Program('add 1 2', 1).eval([1]) == 3

    # More args than max_arity means we can't fill them
    with raises(IndexError):
        assert Program('add $0 $1', 1).eval()

    # Less params than needed is a problem
    with raises(IndexError):
        Program('add 1').eval()

    with raises(IndexError):
        Program('add 1 $0', 1).eval()

    # List representation
    assert Program(('add', '1', '2'), 1).eval([1]) == 3

    # Named columns
    with raises(ValueError):
        Program('$a', 1).eval([1])

    assert Program('$a', 1, columns=['a']).eval([[1, 2, 3]]) == [1, 2, 3]


def test_simplify():
    def assert_simplified(source, target):
        assert str(Program(source, 3).simplify()) == f"Program('{target}', 3)"

    assert_simplified('3.0', '3.0')
    assert_simplified('add 1 2', '3.0')
    assert_simplified('add 0 $1', '$1')
    assert_simplified('add $1 0', '$1')
    assert_simplified('add 0.0 0.0', '0.0')
    assert_simplified('add $1 neg $2', 'sub $1 $2')
    assert_simplified('add neg $1 $2', 'sub $2 $1')
    assert_simplified('sub $a 0.0', '$a')
    assert_simplified('sub 0.0 $a', 'neg $a')
    assert_simplified('add 0.0 sub $a $b', 'sub $a $b')
    assert_simplified('add sub $a $b 0.0', 'sub $a $b')
    assert_simplified('pow rec 0.5 $a', 'pow 2.0 $a')


def test_point_mutate():
    for i in range(10):
        assert Program('3').point_mutation() != Program('3')
        assert Program('3', 2).point_mutation() != Program('3', 2)
        assert Program('$0', 1).point_mutation() != Program('$0', 1)
        assert Program('$1', 2).point_mutation() != Program('$1', 2)
        assert Program('exp 1').point_mutation() != Program('exp 1')
        assert Program('add 2 3').point_mutation() != Program('add 2 3')

        # Manual inspection; check that anything can mutate (even inside)
        # To do this, change the != to ==
        assert Program('add sub 2 3 4').point_mutation() != Program('add sub 2 3 4')


def test_hoist_mutate():
    random.seed(0)
    assert Program('exp 1').hoist_mutation() == Program('1.0')
    assert Program('add 1 1').hoist_mutation() == Program('1.0')
    assert Program('add add 1 1 1').hoist_mutation() == Program('add 1.0 1.0')
    assert Program('add add add 1 1 1 1').hoist_mutation() == Program('1.0')


def test_prune_mutate():
    random.seed(0)
    pruned = Program('exp neg 1').prune_mutation()

    assert len(pruned.source) == 2
    assert pruned.source[0] == 'exp'


def test_crossover():
    random.seed(0)
    c = Configuration(complete_tree_as_new_subtree_chance=1)
    a = Program('exp 1', config=c)
    b = Program('add 2 3')
    assert a.crossover(b) == Program('exp add 2 3')

    c = Configuration(complete_tree_as_new_subtree_chance=0)
    a = Program('exp 1', config=c)
    b = Program('add 2 3')
    assert a.crossover(b) == Program('exp 3.0')


def test_fitness():
    assert fitness(Program('0', 1), [[1]], [[1]]) == (1, 1)
    assert fitness(Program('$0', 1), [[1]], [[1]]) == (0, 1)
    assert fitness(Program('nan', 1), [[1]], [[1]]) == (float('inf'), 1)
    assert fitness(Program('div 1 0', 1), [[1]], [[1]]) == (float('inf'), 3)
    assert fitness(Program('rec add $1 $0', 2), [[0], [0]], [[0]]) ==\
           (float('inf'), 4)


if __name__ == '__main__':
    import pytest

    pytest.main(['test_ga.py', '--color=yes'])
