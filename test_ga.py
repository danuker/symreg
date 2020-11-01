from ga import GA, Program
import numpy as np
import pandas as pd
import pytest


def test_program():
    assert Program('3')._p == [3]
    assert Program('add 3 4')._p == [np.add, 3, 4]
    assert Program('3').eval() == 3
    assert Program('add 3 4').eval() == 7
    assert Program('sub 5 4').eval() == 1
    assert Program('add 3 sub 5 4').eval() == 4
    assert Program('exp 1').eval() == 2.718281828459045
    assert Program('div $0 4', 1).eval(1) == 0.25
    assert Program('div $0 0', 1).eval(1) == float('inf')
    assert Program('div $0 0', 1).eval(-1) == float('-inf')
    assert Program('div $0 -inf', 1).eval(1) == 0.0
    assert {Program('123'), Program('123')} == {Program('123')}
    assert (Program('add $0 $1', 2).eval(np.ones(10), np.zeros(10)) == np.ones(10)).all()
    assert (Program('add $0 $1', 2).eval(pd.Series([0, 1]), pd.Series([1, 2])) == pd.Series([1, 3])).all()
    assert (Program('add $0 $1', 2).eval(pd.DataFrame([[0, 1], [2, 3], [4, 5]])) == pd.Series([1, 5, 9])).all()

    # More args than needed should be no problem
    assert Program('add 1 2', 1).eval(1) == 3

    # More args than max_arity means we can't generate them
    with pytest.raises(ValueError):
        assert Program('add $0 $1', 1).eval()

    # Less params than needed is a problem
    with pytest.raises(ValueError):
        Program('add 1')

    with pytest.raises(IndexError):
        Program('add 1 $0', 1).eval()

    # List representation
    assert Program(('add', '1', '2')).eval(1) == 3


def test_point_mutate():
    for i in range(10):
        assert Program('3').mutate() != Program('3')
        assert Program('$0', 1).mutate() != Program('$0', 1)
        assert Program('$1', 2).mutate() != Program('$1', 2)
        assert Program('exp 1').mutate() != Program('exp 1')
        assert Program('add 2 3').mutate() != Program('add 2 3')


def test_grow_mutate():
    # TODO
    assert len(Program('3').mutate()._source) > 1


# def test_reproduce():
#     ga = GA({Program('a'), Program('b'), Program('c')})
#     children = ga.reproduce()
#     assert len(children) == 3


if __name__ == '__main__':
    import pytest

    pytest.main(['test_ga.py'])

