from nsgaii import fast_non_dominated_sort, crowding_distance_assignment, nsgaii_cull


def test_fast_non_dominated_sort():
    assert fast_non_dominated_sort({}) == \
           {}
    assert fast_non_dominated_sort({'best1': (1, 2), 'best2': (2, 1)}) == \
           {1: {'best1', 'best2'}}
    assert fast_non_dominated_sort({'best1': (1, 2), 'best2': (2, 1), 'worse': (2, 2)}) == \
           {1: {'best1', 'best2'}, 2: {'worse'}}

    assert fast_non_dominated_sort({'worse': (1, 2), 'best_inf': (float('-inf'), 2)}) == \
           {1: {'best_inf'}, 2: {'worse'}}

    assert fast_non_dominated_sort({'worse': (float('nan'), 2), 'best': (1, 2)}) == \
           {1: {'best'}, 2: {'worse'}}

    assert fast_non_dominated_sort({'ugly': (float('nan'), 1), 'good': (1, 2)}) == \
           {1: {'ugly', 'good'}}


def test_crowding_distance_assignment():
    assert crowding_distance_assignment({}) == {}
    assert crowding_distance_assignment({'best1': (1, 2), 'best2': (2, 1)}) == \
           {'best1': float('inf'), 'best2': float('inf')}

    assert crowding_distance_assignment(
        {'best1': (1, 2), 'best2': (2, 1), 'mid': (1.5, 1.5)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': 2}

    # Scaling
    assert crowding_distance_assignment(
        {'best1': (1, 2), 'best2': (3, 3), 'mid': (2, 2.5)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': 2}


def test_nsgaii_cull():
    start = {'best1': (1, 2), 'best2': (2, 1), 'worse': (1, 1), 'worst': (2, 2)}
    end = {'best1': (1, 2), 'best2': (2, 1), 'worse': (1, 1)}
    assert nsgaii_cull(start, 3) == end

    start = {'best1': (1, 2), 'best2': (2, 1), 'not_on_shortlist': (1.5, 1.5)}
    end = {'best1': (1, 2), 'best2': (2, 1)}
    assert nsgaii_cull(start, 2) == end

    # Equidistant
    start = {'best1': (1, 2), 'best2': (2, 1), 'best_equidistant': (1.5, 1.5), 'best_too_close': (1.25, 1.75)}
    assert nsgaii_cull(start, 3) == {'best1': (1, 2), 'best2': (2, 1), 'best_equidistant': (1.5, 1.5)}


if __name__ == '__main__':
    import pytest

    pytest.main(['test_nsgaii.py', '-vv'])