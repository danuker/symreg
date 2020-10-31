from nsgaii import fast_non_dominated_sort, crowding_distance_assignment, nsgaii_cull


def test_fast_non_dominated_sort():
    assert fast_non_dominated_sort({}) == \
           {}
    assert fast_non_dominated_sort({'best1': (0, 1), 'best2': (1, 0)}) == \
           {1: {'best1', 'best2'}}
    assert fast_non_dominated_sort({'best1': (0, 1), 'best2': (1, 0), 'worse': (1, 1)}) == \
           {1: {'best1', 'best2'}, 2: {'worse'}}

    assert fast_non_dominated_sort({'worse': (0, 1), 'best_inf': (float('-inf'), 1)}) == \
           {1: {'best_inf'}, 2: {'worse'}}

    assert fast_non_dominated_sort({'worse': (float('nan'), 1), 'best': (0, 1)}) == \
           {1: {'best'}, 2: {'worse'}}


def test_crowding_distance_assignment():
    assert crowding_distance_assignment({}) == {}
    assert crowding_distance_assignment({'best1': (0, 1), 'best2': (1, 0)}) == \
           {'best1': float('inf'), 'best2': float('inf')}

    assert crowding_distance_assignment(
        {'best1': (0, 1), 'best2': (1, 0), 'mid': (0.5, 0.5)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': 2}

    # Scaling
    assert crowding_distance_assignment(
        {'best1': (0, 1), 'best2': (1, 0), 'mid': (0.5, 0.5)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': 2}


def test_nsgaii_cull():
    input = {'best1': (0, 1), 'best2': (1, 0), 'worse': (1, 1), 'worst': (2, 2)}
    output = {'best1': (0, 1), 'best2': (1, 0), 'worse': (1, 1)}
    assert nsgaii_cull(input, 3) == output

    input = {'best1': (0, 1), 'best2': (1, 0), 'not_on_shortlist': (0.5, 0.5)}
    output = {'best1': (0, 1), 'best2': (1, 0)}
    assert nsgaii_cull(input, 2) == output

    # Equidistant
    input = {'best1': (0, 1), 'best2': (1, 0), 'best_equidistant': (0.5, 0.5), 'best_too_close': (0.25, 0.75)}
    assert nsgaii_cull(input, 3) == {'best1': (0, 1), 'best2': (1, 0), 'best_equidistant': (0.5, 0.5)}


if __name__ == '__main__':
    import pytest

    pytest.main(['test_nsgaii.py', '-vv'])