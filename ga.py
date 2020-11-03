from collections import defaultdict
import random

import numpy as np

from nsgaii import nsgaii_cull

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
opnames = {v[0]: opname for opname, v in blocks.items()}


def set_choice(s):
    return random.choice(list(s))


def _eval_block(token: str):
    try:
        return float(token), 0
    except ValueError:
        if token.startswith('$'):
            return token, 0
        return blocks[token]


class Program:
    """ Lisp without the brackets, but with reproduction """

    def __init__(self, source, max_arity=0):
        if isinstance(source, str):
            self.source = source.split(' ')
        else:
            self.source = source

        self._max_arity = max_arity
        self._p = self._parse_source()

    def _parse_source(self):
        p = []
        check_arity = 1
        for tok in self.source:
            block, arity = _eval_block(tok)

            if check_arity > 0:
                p.append(block)
                check_arity += arity - 1
        if check_arity > 0:
            raise ValueError(f'Invalid program; arity left to fill: {check_arity}')
        return p

    def _from_df(self, X):
        """Transpose X matrix into columns as args"""

        args = np.array(X, ndmin=2)

        if len(args.shape) > 2:
            raise ValueError(f'Invalid args shape: {args.shape}')

        if args.shape[1] != self._max_arity:
            raise ValueError(
                f'Args must have {self._max_arity} cols, but has {args.shape[1]}'
            )
        return args.transpose()

    def eval(self, X=()):
        """Run program on args, computing output"""
        args = self._from_df(X)
        stack = self._p[:]
        ops = {v[0] for v in blocks.values()}
        result_stack = ()

        while stack:
            op = stack.pop()
            if op in ops:
                opname = opnames[op]
                arity = arities[opname]
                my_args = result_stack[:arity]
                result_stack = (op(*my_args),) + result_stack[arity:]
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
        return random.choice([
            self.point_mutation,
            self.point_mutation,
            self.point_mutation,
            self.grow_mutation,
        ])()

    def grow_mutation(self):
        """Replace a leaf with an op and two leaves"""
        i = 0
        while self.source[i] in blocks:
            i = random.randrange(len(self.source))

        new_op = set_choice(blocks)
        needed_args = blocks[new_op][1]
        new_args = [self._new_leaf('0')] * needed_args
        if random.choice([True, False]):
            new_args[random.randrange(len(new_args))] = self.source[i]

        new_source = self.source[:i] \
                     + [new_op] \
                     + new_args \
                     + self.source[i + 1:]

        return Program(new_source, self._max_arity)

    def point_mutation(self):
        """Replace a source op while preserving all arities"""
        i = random.randrange(len(self.source))  # Index of source block to mutate
        op = self.source[i]
        max_arity = self._max_arity
        candidates = self._ops_with_same_arity(op)
        if not candidates:
            # We must choose a constant or a parameter
            chosen = self._new_leaf(op)
        else:
            chosen = random.choice(list(candidates))
        new_source = self.source[:i] + [chosen] + self.source[i + 1:]
        return Program(new_source, self._max_arity)

    @staticmethod
    def _ops_with_same_arity(op):
        return set(name for name in blocks if arities[op] == arities[name]) - {op}

    def _new_leaf(self, op):
        choices = set()

        if not op.startswith('$'):
            choices.add(str(float(op) + random.gauss(0, 1)))
        else:
            choices.add(str(random.gauss(0, 1)))

        if self._max_arity:
            choices.add(f'${random.randrange(self._max_arity)}')

        return set_choice(choices - {op})

    def __repr__(self):
        return f"Program('{' '.join(self.source)}', {self._max_arity})"

    def __eq__(self, other):
        return self.source == other.source

    def __hash__(self):
        return hash(self.__repr__())

    def _swap(self, ith):
        """Swap a program part with another of the same arity"""
        assert 0 <= ith < len(self.source)

    @classmethod
    def random(cls, max_arity):
        return Program('0', max_arity).mutate()


class GA:
    def __init__(self, n=20, steps=20):
        self.individuals = set()
        self.n = n
        self.steps = steps
        self.steps_taken = 0
        self.old_scores = dict()

    def predict(self, X):
        """Predict using the individual with the least training error"""
        errors = ((v[0], i) for i, v in self.old_scores.items())
        best, _ = min(errors, key=lambda e: e[0])
        return list(self.old_scores)[0].eval(X)

    def fit(self, X, y):
        self.individuals = set(
            Program.random(max_arity=len(X[0]))
            for _ in range(self.n)
        )
        self.old_scores = dict()
        self.steps_taken = 0
        for i in range(self.steps):
            self._step(X, y)

    def fit_partial(self, X, y):
        for i in range(self.steps):
            self._step(X, y)


    def _step(self, X, y):
        new_gen = {i.mutate() for i in self.individuals}
        new_scores = {i: self._fitness(i, X, y) for i in new_gen if i not in self.old_scores}
        self.old_scores.update(new_scores)
        final = nsgaii_cull(self.old_scores, self.n)
        self.steps_taken += 1
        self.old_scores = final
        self.individuals = set(final.keys())

    def _fitness(self, program: Program, X, y):
        complexity = len(program.source)
        try:
            diff = (y - program.eval(X))
            score = np.average(diff * diff)
        except ValueError:
            score = float('inf')
        return score, complexity


if __name__ == '__main__':
    import pytest

    pytest.main(['test_ga.py', '--color=yes'])
