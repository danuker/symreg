from collections import defaultdict
import random

import numpy as np
import pandas as pd

np.seterr(divide='ignore')  # We know we will get +/- infs

blocks = {
    # name: (function, arity)
    'add': (np.add, 2),
    'sub': (np.subtract, 2),
    'mul': (np.multiply, 2),
    'div': (np.divide, 2),
    'pow': (np.power, 2),

    'exp': (np.exp, 1),
    'neg': (np.negative, 1)

    # Arity 0: args ($0, $1...) and constants
}

arities = defaultdict(lambda: 0, {name: v[1] for name, v in blocks.items()})


def _eval_block(token: str):
    try:
        return float(token), 0
    except ValueError:
        if token.startswith('$'):
            return token, 0
        return blocks[token]


def _from_df(X):
    """Unpack X matrix into columns as args"""
    if len(X) == 1 and isinstance(X[0], pd.Series):
        # Interpret Pandas series as column (1 random variable, many samples)
        return X[0],
    elif len(X) == 1 and isinstance(X[0], pd.DataFrame):
        return tuple(X[0][c] for c in X[0])
    else:
        X = np.array(X)
        return tuple(X.transpose())


class Program:
    """ Lisp without the brackets, but with reproduction """

    def __init__(self, source, max_arity=0):
        if isinstance(source, str):
            self._source = source.split(' ')
        else:
            self._source = source

        self._max_arity = max_arity
        self._p = self._parse_source()

    def _parse_source(self):
        p = []
        check_arity = 1
        for tok in self._source:
            block, arity = _eval_block(tok)

            if check_arity > 0:
                p.append(block)
                check_arity += arity - 1
        if check_arity > 0:
            raise ValueError(f'Invalid program; arity left to fill: {check_arity}')
        return p

    def eval(self, *args):
        """Run program on args, computing output"""
        args = _from_df(args)
        if len(args) != self._max_arity:
            raise ValueError(
                f'Wrong number of params/columns: was {len(args)}, expected {self._max_arity}'
            )
        stack = self._p[:]
        ops = {v[0] for v in blocks.values()}
        result_stack = ()

        while stack:
            op = stack.pop()
            if op in ops:
                result_stack = (op(*result_stack),)
            elif isinstance(op, str):
                argnum = int(op[1:])
                if argnum >= self._max_arity:
                    raise ValueError(f'Bad max_arity; must be at least {argnum + 1}')
                result_stack = (args[argnum],) + result_stack
            else:
                result_stack = (op,) + result_stack

        assert len(result_stack) == 1
        return np.array(result_stack[0]).transpose()

    def mutate(self):
        """Replace a source op while preserving all arities"""

        i = random.randrange(len(self._source))  # Index of source block to mutate
        op = self._source[i]
        max_arity = self._max_arity
        candidates = self._ops_with_same_arity(op)

        if not candidates:
            # We must choose a constant or a parameter
            chosen = self._mutate_leaf(max_arity, op)
        else:
            chosen = random.choice(list(candidates))

        new_source = self._source[:i] + [chosen] + self._source[i + 1:]

        return Program(new_source, self._max_arity)

    @staticmethod
    def _ops_with_same_arity(op):
        return set(name for name in blocks if arities[op] == arities[name]) - {op}

    @staticmethod
    def _mutate_leaf(max_arity, op):
        if not op.startswith('$'):
            random_float = str(float(op) + random.gauss(0, 1))
        else:
            random_float = str(random.gauss(0, 1))

        if max_arity >= 2:
            # We can only choose a different param if we have at least 2
            op_n = int(op[1:])
            random_param = f'${random.choice(list(set(range(max_arity)) - {op_n}))}'
        else:
            random_param = random_float

        return random.choice([random_float, random_float, random_param])

    def __repr__(self):
        return f"Program('{' '.join(self._source)}')"

    def __eq__(self, other):
        return self._source == other._source

    def __hash__(self):
        return hash(self.__repr__())

    def _swap(self, ith):
        """Swap a program part with another of the same arity"""
        assert 0 <= ith < len(self._source)


class GA:
    def __init__(self, individuals: set):
        self.individuals = set(individuals)

    def reproduce(self):
        return self.individuals.copy()

    def evaluate(self, X):
        raw_results = {
            i: i.eval(X) for i in self.individuals
        }

        def _broadcast(v):
            if isinstance(v, np.ndarray):
                return v
            else:
                return np.full((len(X), 1), v)

        return {
            k: _broadcast(v) for k, v in raw_results.items()
        }



if __name__ == '__main__':
    import pytest

    pytest.main(['test_ga.py', '--color=yes'])
