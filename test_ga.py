from ga import GA, Program
import numpy as np
import pandas as pd
from pytest import raises


def test_program():
    # assert Program('3')._p == [3]
    # assert Program('add 3 4')._p == [np.add, 3, 4]
    # assert Program('3').eval() == 3
    # assert Program('add 3 4').eval() == 7
    # assert Program('sub 5 4').eval() == 1
    # assert Program('add add 1 2 3').eval() == 6
    assert Program('sub $0 neg -1', 1).eval([[1]]) == 0
    assert Program('sub neg -1 $0', 1).eval([[-1]]) == 2
    assert Program('add 3 sub 5 4').eval() == 4
    assert Program('exp 1').eval() == 2.718281828459045
    assert Program('div $0 $1', 2).eval([-1, 2]) == float(-0.5)
    assert Program('div $0 4', 1).eval(1) == 0.25
    assert Program('div $0 0', 1).eval(1) == float('inf')
    assert Program('div $0 0', 1).eval(-1) == float('-inf')
    assert Program('div $0 -inf', 1).eval(1) == 0.0
    assert {Program('123'), Program('123')} == {Program('123')}
    assert (Program('add $0 1', 1).eval(np.array([[0], [1]])) == [1, 2]).all()
    assert (Program('add $0 $1', 2).eval(np.array([[0, 1], [2, 3]])) == np.array([1, 5])).all()
    assert (Program('add $0 $1', 2).eval(pd.DataFrame([[0, 1], [2, 3], [4, 5]]))
            == pd.Series([1, 5, 9])).all()
    assert (Program('add $0 1', 1).eval([[0], [1]]) == pd.Series([1, 2])).all()

    assert (Program('$1', 2).eval(pd.DataFrame([[0, 0], [0, 1], [0, 2]])) == pd.Series([0, 1, 2])).all()

    # More args than needed should be no problem
    assert Program('add 1 2', 1).eval(1) == 3

    # More args than max_arity means we can't fill them
    with raises(ValueError):
        assert Program('add $0 $1', 1).eval()

    # Less params than needed is a problem
    with raises(ValueError):
        Program('add 1')

    with raises(ValueError):
        Program('add 1 $0', 1).eval()

    # List representation
    assert Program(('add', '1', '2'), 1).eval(1) == 3


def test_point_mutate():
    for i in range(10):
        assert Program('3').point_mutation() != Program('3')
        assert Program('3', 2).point_mutation() != Program('3', 2)
        assert Program('$0', 1).point_mutation() != Program('$0', 1)
        assert Program('$1', 2).point_mutation() != Program('$1', 2)
        assert Program('exp 1').point_mutation() != Program('exp 1')
        assert Program('add 2 3').point_mutation() != Program('add 2 3')


def test_grow_mutate():
    assert len(Program('3').grow_mutation().source) > 1


if __name__ == '__main__':
    import pytest

    pytest.main(['test_ga.py', '--color=yes'])
