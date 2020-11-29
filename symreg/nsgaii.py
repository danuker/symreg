from collections import defaultdict
from dataclasses import dataclass

from orderedset import OrderedSet


@dataclass(frozen=True)
class SolutionScore:
    """And individual together with its scores, used for fast nondominated sort"""
    individual: object
    scores: tuple

    @classmethod
    def scores_from_dict(cls, score_dict: dict):
        return OrderedSet(
            SolutionScore(p, p_scores) for p, p_scores in score_dict.items())

    def dominates(self, other):
        """Smaller error/complexity is better"""
        at_least_as_good = all(s <= o for s, o in zip(self.scores, other.scores))
        better_in_some_respect = any(s < o for s, o in zip(self.scores, other.scores))

        return at_least_as_good and better_in_some_respect


def fast_non_dominated_sort(scores: dict, sort=None) -> dict:
    """
        Group individuals by rank of their Pareto front
        :param scores: {individual -> (score1, score2, ...)}
        :param sort: '2d' for cheap, 'nd' for general
        :returns {individual: Pareto rank}
    """

    scores = SolutionScore.scores_from_dict(scores)
    try:
        dims = len(list(scores)[0].scores)
    except IndexError:
        dims = 2

    if sort == '2d' and dims != 2:
        raise ValueError(f'2D sort is not possible. Data has {dims} dimensions.')

    if sort is None:
        sort = '2d' if dims == 2 else 'nd'

    if sort == '2d':
        fronts = _2dim_pareto_ranking(scores)
    elif sort == 'nd':
        fronts = ndim_pareto_ranking(scores)
    else:
        raise ValueError(f'Bad sort: {sort}')

    return {
        key: OrderedSet(v.individual for v in value)
        for key, value in fronts.items() if value
    }


def _get_2d_front(sol_scores):
    lexicographic = sorted(sol_scores, key=lambda ss: ss.scores)
    min_y_seen = float('inf')
    front = []
    for s in lexicographic:
        if s.scores[1] < min_y_seen:
            min_y_seen = s.scores[1]
            front.append(s)

    return front


def _2dim_pareto_ranking(sol_scores):
    fronts = {}
    i = 0
    remaining = sol_scores
    while remaining:
        i += 1
        fronts[i] = _get_2d_front(remaining)
        remaining.difference_update(fronts[i])

    return fronts


def ndim_pareto_ranking(scores):
    S = defaultdict(OrderedSet)  # p is superior to individuals in S[p]
    n = defaultdict(lambda: 0)  # p is dominated by n[p] individuals
    fronts = defaultdict(OrderedSet)  # individuals in front 1 are fronts[1]

    # Create domination map
    for p in scores:
        for q in scores - {p}:
            if p.dominates(q):
                S[p].update({q})
            elif q.dominates(p):
                n[p] += 1

        if n[p] == 0:
            fronts[1].update({p})
    # Iteratively eliminate current Pareto frontier, and find members of next best
    i = 1
    while fronts[i]:
        Q = OrderedSet()  # Next front

        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.update({q})
        i += 1
        fronts[i] = Q
    return fronts


def _peek_any(set_or_dict):
    return list(set_or_dict)[0]


def crowding_distance_assignment(scores: dict) -> dict:
    """
        Calculate crowding

        * Implemented as dimension-normalized distance between
        the 2 surrounding neighbors from given Pareto front
        * Tries to stay close to the paper's pseudocode
        (albeit it is quite long and imperative-style)

        :param scores: {individual -> (score1, score2, ...)}
        :returns {individual: distance}
    """

    distance = {i: 0 for i in scores}

    try:
        M = len(_peek_any(scores.values()))  # number of objectives
        for m in range(M):
            # Sort using each objective value
            inds = tuple(sorted(scores.keys(), key=lambda i: scores[i][m]))

            fm_min, fm_max = scores[inds[0]][m], scores[inds[-1]][m]

            for i in inds:
                if scores[i][m] in [fm_min, fm_max]:
                    distance[i] = float('inf')

            for i in range(1, len(inds) - 1):
                earlier = scores[inds[i + 1]]
                later = scores[inds[i - 1]]
                space = (earlier[m] - later[m])
                normalized = (fm_max - fm_min)
                try:
                    distance[inds[i]] += space / normalized
                except ZeroDivisionError:
                    distance[inds[i]] = float('inf')

    except IndexError:
        # _peek_any failed: we don't have any individuals
        pass

    return distance


def nsgaii_cull(start_pop, n_out, sort=None):
    """
    Remove individuals that are not fit enough according to NSGA-II

    :param start_pop:   {individual -> (score1, score2, ...), ...}
    :param n_out:       number of individuals to survive
    :param sort:        '2d' for cheap 2d, 'nd' for general
    :return:            {individual -> (score1, score2, ...), ...}
    """
    pareto_front = fast_non_dominated_sort(start_pop, sort)
    crowding_distance = crowding_distance_assignment(start_pop)
    end_pop = []
    for front in sorted(pareto_front.keys()):
        sorted_front = sorted(pareto_front[front], key=lambda i: -crowding_distance[i])
        end_pop.extend(sorted_front)

        if len(end_pop) > n_out:
            break

    flat = {i: start_pop[i] for i in end_pop[:n_out]}
    if 1 not in pareto_front:
        raise ValueError(f'Invalid pareto front for {start_pop}')
    return flat, pareto_front[1]
