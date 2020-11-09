import math
from collections import defaultdict
import random
import numpy as np

from .nsgaii import nsgaii_cull

np.seterr(all='ignore')  # We know we will get numerical funny business

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


def fitness(program, Xt, y):
    complexity = len(program.source)
    try:
        diff = (y - program.eval(Xt))
        error = np.average(diff * diff)
        if error < 0 or math.isnan(error):
            return float('inf'), complexity
        return error, complexity
    except ValueError:
        return float('inf'), complexity


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
    """ Lisp without the brackets, but more... alive """

    def __init__(
            self,
            source,
            max_arity=0,
            zero_program_chance=0.5,
            grow_root_mutation_chance=.2,
            grow_leaf_mutation_chance=.4,
            int_std=4,
            float_std=5,
    ):
        self._max_arity = max_arity
        self.zero_program_chance = zero_program_chance
        self.grow_root_mutation_chance = grow_root_mutation_chance
        self.grow_leaf_mutation_chance = grow_leaf_mutation_chance
        self.int_std = int_std
        self.float_std = float_std
        if isinstance(source, str):
            self.source = tuple(source.split(' '))
        else:
            self.source = tuple(source)
        self._p = self._parse_source()

    def from_source(self, new_source):
        return Program(
            tuple(new_source),
            max_arity=self._max_arity,
            zero_program_chance=self.zero_program_chance,
            grow_root_mutation_chance=self.grow_root_mutation_chance,
            grow_leaf_mutation_chance=self.grow_leaf_mutation_chance,
            int_std=self.int_std,
            float_std=self.float_std,
        )

    def _parse_source(self):
        p = []
        check_arity = 1
        for tok in self.source:
            block, arity = _eval_block(tok)

            if check_arity > 0:
                p.append(block)
                check_arity += arity - 1
            else:
                raise ValueError(f'Too much program')
        if check_arity != 0:
            raise ValueError(f'Invalid program; arity left to fill: {check_arity}')
        return p

    def eval(self, args=()):
        """
        Run program on args, computing output
        Args: 1 row = 1 parameter
        """
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
        assert self.grow_leaf_mutation_chance + self.grow_root_mutation_chance < 1
        choice = random.random()
        if choice < self.grow_leaf_mutation_chance:
            return self.grow_leaf_mutation()
        elif choice < self.grow_leaf_mutation_chance + self.grow_root_mutation_chance:
            return self.grow_root_mutation()
        else:
            return self.point_mutation()

    def grow_leaf_mutation(self):
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
                     + (new_op,) \
                     + tuple(new_args) \
                     + self.source[i + 1:]

        # print('leaf', self.source, new_source)
        return self.from_source(new_source)

    def grow_root_mutation(self):
        """Make a new root, and make the old root an arg"""
        new_op = set_choice(blocks)
        needed_args = blocks[new_op][1]

        new_args = (self._new_leaf('0'),) * needed_args
        i = random.randrange(len(new_args))
        new_source = (new_op,) + new_args[:i] + self.source + new_args[(i + 1):]

        return self.from_source(new_source)

    def point_mutation(self):
        """Replace a source op while preserving all arities"""
        i = random.randrange(len(self.source))  # Index of source block to mutate
        op = self.source[i]
        candidates = self._ops_with_same_arity(op)
        if not candidates:
            # We must choose a constant or a parameter
            chosen = self._new_leaf(op)
        else:
            chosen = random.choice(list(candidates))
        new_source = self.source[:i] + (chosen,) + self.source[i + 1:]

        return self.from_source(new_source)

    @staticmethod
    def _ops_with_same_arity(op):
        return set(name for name in blocks if arities[op] == arities[name]) - {op}

    def _new_leaf(self, op):
        choices = set()

        if not op.startswith('$'):
            choices.add(str(float(op) + random.gauss(0, self.float_std)))
        else:
            choices.add(str(random.gauss(0, self.float_std)))
        choices.add(str(int(random.gauss(0, self.int_std))))

        if self._max_arity:
            choices.add(f'${random.randrange(self._max_arity)}')

        return set_choice(choices - {op})

    def __repr__(self):
        return f"Program('{' '.join(self.source)}', {self._max_arity})"

    def __eq__(self, other):
        return self.source == other.source

    def __hash__(self):
        return hash((self.source, self._max_arity))

    def _swap(self, ith):
        """Swap a program part with another of the same arity"""
        assert 0 <= ith < len(self.source)


class GA:
    def __init__(self, n, steps,
                 zero_program_chance,
                 grow_leaf_mutation_chance,
                 grow_root_mutation_chance,
                 int_std,
                 float_std,
                 ):
        self.n = n
        self.steps = steps
        self.zero_program_chance = zero_program_chance
        self.grow_leaf_mutation_chance = grow_leaf_mutation_chance
        self.grow_root_mutation_chance = grow_root_mutation_chance
        self.int_std = int_std
        self.float_std = float_std

        self.individuals = set()
        self.steps_taken = 0
        self._max_arity = None
        self.old_scores = dict()

    def _from_df(self, X):
        """Transpose the X array into columns as args"""
        Xa = np.array(X, ndmin=2)

        if len(Xa.shape) > 2:
            raise ValueError(f'Invalid args shape: {Xa.shape}')

        if self._max_arity is not None and Xa.shape[1] != self._max_arity:
            raise ValueError(
                f'Args must have {self._max_arity} cols, but has {Xa.shape[1]}'
            )
        return Xa.transpose()

    def predict(self, X):
        """Predict using the individual with the least training error"""
        errors = ((v[0], i) for i, v in self.old_scores.items())
        _, best = min(errors, key=lambda e: e[0])
        return best.eval(self._from_df(X))

    def fit(self, X, y):
        params = self._from_df(X)
        self._max_arity = len(params)

        p = Program(
            source='0',
            max_arity=self._max_arity,
            grow_root_mutation_chance=self.grow_root_mutation_chance,
            grow_leaf_mutation_chance=self.grow_leaf_mutation_chance,
            int_std=self.int_std,
            float_std=self.float_std,
        )

        self.individuals = set(
            p if random.random() < self.zero_program_chance else p.mutate()
            for _ in range(self.n)
        )
        self.old_scores = dict()
        self.steps_taken = 0
        for i in range(self.steps):
            self._step(params, y)

    def fit_partial(self, X, y):
        Xt = (self._from_df(X))
        for i in range(self.steps):
            self._step(Xt, y)

    def _step(self, Xt, y):
        new_gen = {i.mutate() for i in self.individuals}
        new_scores = {i: fitness(i, Xt, y) for i in new_gen if i not in self.old_scores}
        self.old_scores.update(new_scores)
        final = nsgaii_cull(self.old_scores, self.n)
        self.steps_taken += 1
        self.old_scores = final
        self.individuals = set(final.keys())

