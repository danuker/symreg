from orderedset import OrderedSet

from symreg.nsgaii import fast_non_dominated_sort, crowding_distance_assignment, nsgaii_cull


def test_fast_non_dominated_sort():
    for sort in ['2d', 'nd']:
        def sorter(scores):
            return fast_non_dominated_sort(scores, 1e9, sort)

        assert sorter({}) == {}
        assert sorter({'best1': (1, 2), 'best2': (2, 1)}) == \
               {1: {'best1', 'best2'}}
        assert sorter({'best1': (1, 2), 'best2': (2, 1), 'worse': (2, 2)}) == \
               {1: {'best1', 'best2'}, 2: {'worse'}}

        assert sorter({'worse': (1, 2), 'best_inf': (float('-inf'), 2)}) == \
               {1: {'best_inf'}, 2: {'worse'}}

        assert sorter({'worse': (float('inf'), 2), 'best': (1, 2)}) == \
               {1: {'best'}, 2: {'worse'}}


def test_crowding_distance_assignment():
    assert crowding_distance_assignment({}) == {}
    assert crowding_distance_assignment({'best1': (1, 2), 'best2': (2, 1)}) == \
           {'best1': float('inf'), 'best2': float('inf')}

    assert crowding_distance_assignment(
        {'best1': (1, 2), 'best2': (2, 1), 'mid': (1.5, 1.5)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': 2}

    assert crowding_distance_assignment(
        {'best1': (1, 2), 'best2': (2, 1), 'min': (1, 1), 'max': (1, 1)}
    ) == {
        'best1': float('inf'),
        'best2': float('inf'),
        'min': float('inf'),
        'max': float('inf'),
    }

    # Scaling
    assert crowding_distance_assignment(
        {'best1': (1, 2), 'best2': (3, 3), 'mid': (2, 2.5)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': 2}

    assert crowding_distance_assignment(
        {'best1': (1, 2), 'best2': (3, 2), 'mid': (2, 2)}
    ) == {'best1': float('inf'), 'best2': float('inf'), 'mid': float('inf')}


def test_nsgaii_cull():
    for sort in ['2d', 'nd']:
        start = {'best1': (1, 2), 'best2': (2, 1), 'bestest': (1, 1), 'worst': (2, 2)}
        end = {'best1': (1, 2), 'best2': (2, 1), 'bestest': (1, 1)}
        pareto = OrderedSet(['bestest'])
        assert nsgaii_cull(start, 3, sort) == (end, pareto)

        start = {'best1': (1, 2), 'best2': (2, 1), 'not_on_shortlist': (1.5, 1.5)}
        assert nsgaii_cull(start, 2, sort)[0].keys() == {'best1', 'best2'}

        # Equidistant
        start = {'best1': (1, 2), 'best2': (2, 1), 'best_equidistant': (1.5, 1.5), 'best_too_close': (1.25, 1.75)}
        assert nsgaii_cull(start, 3, sort)[0] == {'best1': (1, 2), 'best2': (2, 1), 'best_equidistant': (1.5, 1.5)}

        # Discard along axes
        start = {
            'p_bad': (0, 100000),
            'p': (0, 1),
            'p1_pareto_but_not_convex_hull':  (.4, .7),
            'mid': (.5, .5),
            'p2_pareto_but_not_convex_hull':  (.7, .4),
            'q': (1, 0),
            'q_bad': (10000, 0),
        }
        assert nsgaii_cull(start, 5, sort)[0].keys() == {
            'p', 'mid', 'q',
            'p1_pareto_but_not_convex_hull',
            'p2_pareto_but_not_convex_hull',
        }

        # Discard almost along axes
        start = {
            'p_bad': (0.00001, 100000),
            'p': (0, 1),
            'p1_pareto_but_not_convex_hull':  (.4, .7),
            'mid': (.5, .5),
            'p2_pareto_but_not_convex_hull':  (.7, .4),
            'q': (1, 0),
            'q_bad': (10000, 0.00001),
        }
        assert nsgaii_cull(start, 5, sort)[0].keys() == {
            'p', 'mid', 'q',
            'p1_pareto_but_not_convex_hull',
            'p2_pareto_but_not_convex_hull',
        }


if __name__ == '__main__':
    import pytest

    pytest.main(['test_nsgaii.py', '--color=yes', '-v'])
