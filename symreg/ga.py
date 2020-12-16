import math
from collections import defaultdict
import random
from itertools import chain
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
    'log': (np.log, 1),
    'neg': (np.negative, 1),
    'rec': (np.reciprocal, 1),

    # Arity 0: args ($0, $1...) and constants
}

arities = defaultdict(lambda: 0, {name: v[1] for name, v in blocks.items()})
opnames = {v[0]: opname for opname, v in blocks.items()}
ops_from_name = {opname: v[0] for opname, v in blocks.items()}


def fitness(program, Xt, y):
    complexity = len(program.source)
    try:
        y_est = program.eval(Xt)
        diff = np.subtract(y, y_est)
        error = np.sqrt(np.average(np.square(diff)))
        if error < 0 or math.isnan(error):
            return float('inf'), complexity
        return error, complexity
    except ValueError:
        return float('inf'), complexity


def set_choice(s):
    return random.choice(tuple(s))


def is_elementary(obj):
    return not isinstance(obj, tuple)


def is_constant(obj):
    return not isinstance(obj, (tuple, str))


def _eq_array(a, b):
    res = a == b
    try:
        return bool(res)
    except ValueError:
        # Darn numpy floats cast even tuples to arrays
        return all(res)


def _eval_block(token: str):
    try:
        return float(token), 0
    except ValueError:
        if token.startswith('$'):
            return token, 0
        return blocks[token]


def _ops_with_same_arity(op):
    return tuple(
        name for name in blocks
        if arities[op] == arities[name] and name != op
    )


class Program:
    """ Lisp without the brackets, but more... alive """

    def __init__(
            self,
            source,
            max_arity=0,
            columns=(),
            config=None,
    ):
        if config:
            self.conf = config
        else:
            # Get defaults from Regressor
            from .regressor import Configuration
            self.conf = Configuration()

        self._max_arity = max_arity
        self.columns = tuple(columns)
        self.source = source

        self.optimizers = {
            'add': self.optimize_add,
            'sub': self.optimize_sub,
        }

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        source = self._to_tuple(source)

        self._source = source
        self._p, remaining = self._from_source()
        if remaining:
            raise ValueError(f'Program contains more than one expression: {remaining}')

    @staticmethod
    def _to_tuple(source):
        if isinstance(source, str):
            source = tuple(source.split(' '))
        else:
            source = tuple(source)
        return source

    def from_source(self, new_source):
        return Program(
            tuple(new_source),
            max_arity=self._max_arity,
            columns=self.columns,
            config=self.conf,
        )

    def _from_source(self, source=None) -> tuple:
        """
        Program representation:
        (np.add, (np.divide, 3, 2), 4)

        """
        if source is None:
            source = self._source

        if not source:
            return (), ()

        first, arity = _eval_block(source[0])
        rest = source[1:]
        if not arity:
            is_var = isinstance(first, str) and first[0] == '$'
            if self.columns and is_var:
                try:
                    return f'${self.columns.index(first[1:])}', rest
                except ValueError:
                    pass
            return first, rest

        args = ()
        while len(args) < arity:
            newarg, rest = self._from_source(rest)
            args = args + (newarg,)
        return (first,) + args, rest

    def eval(self, args=()):
        """
        Run program on args, computing output
        Args: 1 row = 1 parameter
        """
        return self._eval(self._p, args)

    def _eval(self, program, args):
        if is_elementary(program):
            if isinstance(program, str):
                if program[0] == '$':
                    try:
                        return args[int(program[1:])]
                    except ValueError:
                        raise ValueError(
                            f'Columns not provided, but named arg used: {program}'
                        )
            return [program]
        evaldargs = tuple(self._eval(p, args) for p in program[1:])

        func = program[0]
        funcname = opnames[func]
        if len(evaldargs) != arities[funcname]:
            raise ValueError(f'Arity mismatch. '
                             f' {funcname} expects {arities[funcname]} args but got: {evaldargs}')

        return func(*evaldargs)

    def mutate(self):
        chances = {
            self.conf.hoist_mutation_chance: self.hoist_mutation,
            self.conf.grow_leaf_mutation_chance: self.grow_leaf_mutation,
            self.conf.grow_root_mutation_chance: self.grow_root_mutation,
            self.conf.prune_mutation_chance: self.prune_mutation,
        }
        assert sum(chances) <= 1

        choice = random.random()

        for c in chances:
            if choice < c:
                return chances[c]()
            choice -= c
        return self.point_mutation()

    def point_mutation(self):
        """Replace a source op while preserving all arities"""
        i = random.randrange(len(self._source))  # Index of source block to mutate
        op = self._source[i]
        candidates = _ops_with_same_arity(op)
        if not candidates:
            # We must choose a constant or a parameter
            chosen = self._new_leaf(op)
        else:
            chosen = set_choice(candidates)
        new_source = self._source[:i] + (chosen,) + self._source[i + 1:]

        return self.from_source(new_source)

    def grow_leaf_mutation(self):
        """Replace a leaf with an op and two leaves"""
        i = 0
        while self._source[i] in blocks:
            i = random.randrange(len(self._source))

        new_op = set_choice(blocks)
        needed_args = blocks[new_op][1]
        new_args = [self._new_leaf('0')] * needed_args
        if random.random() < 0.5:
            new_args[random.randrange(len(new_args))] = self._source[i]

        new_source = \
            self._source[:i] \
            + (new_op,) \
            + tuple(new_args) \
            + self._source[i + 1:]

        return self.from_source(new_source)

    def grow_root_mutation(self):
        """Make a new root, and make the old root an arg"""
        new_op = set_choice(blocks)
        needed_args = blocks[new_op][1]

        new_args = (self._new_leaf('0'),) * needed_args
        i = random.randrange(len(new_args))
        new_source = (new_op,) + new_args[:i] + self._source + new_args[(i + 1):]

        return self.from_source(new_source)

    def hoist_mutation(self):
        """Return a random subtree"""
        if len(self._source) < 2:
            return self.mutate()

        i = random.randrange(len(self._source) - 1) + 1
        parsed, _ = self._subtree_starting_on_index(i)
        return self.from_source(self._to_source(parsed))

    def prune_mutation(self):
        return self.crossover(self.from_source('0').point_mutation())

    def _subtree_starting_on_index(self, i):
        return self._from_source(self._source[i:])

    def crossover(self, other):
        if len(self.source) < 2:
            return self

        if random.random() < self.conf.complete_tree_as_new_subtree_chance:
            new_subtree = other
        else:
            new_subtree = other.hoist_mutation()

        i = random.randrange(len(self._source) - 1) + 1
        parsed, remaining = self._subtree_starting_on_index(i)
        return self.from_source(
            self.source[:i] + new_subtree.source + remaining
        )

    def crossover_with_one(self, many):
        return self.crossover(random.choice(many))

    def _to_source(self, tree) -> tuple:
        if is_elementary(tree):
            return str(tree),
        else:
            funcname = opnames[tree[0]]
            return (funcname,) + tuple(p for arg in tree[1:] for p in self._to_source(arg))

    def _new_leaf(self, op):
        choices = []

        if not op.startswith('$'):
            choices.append(str(float(op) + random.gauss(0, self.conf.float_std)))
        else:
            choices.append(str(random.gauss(0, self.conf.float_std)))
        choices.append(str(float(int(random.gauss(0, self.conf.int_std)))))

        if self._max_arity:
            choices.append(f'${random.randrange(self._max_arity)}')

        return set_choice(c for c in choices if c != op)

    def __repr__(self):
        s = f"Program('{' '.join(self._source)}', {self._max_arity})"
        for i, col in enumerate(self.columns):
            s = s.replace(f'${i}', f'${col}')
        return s

    @staticmethod
    def optimize_add(args):
        if not is_elementary(args[1]):
            if opnames[args[1][0]] == 'neg':
                return ops_from_name['sub'], args[0], args[1][1]

        if not is_elementary(args[0]):
            if opnames[args[0][0]] == 'neg':
                return ops_from_name['sub'], args[1], args[0][1]

        if args[0] == 0.0:
            return args[1]

        if args[1] == 0.0:
            return args[0]

    @staticmethod
    def optimize_sub(args):
        if args[1] == 0:
            return args[0]
        if args[0] == 0:
            return ops_from_name['neg'], args[1]

    def _simplify_tree(self, tree):
        if is_elementary(tree):
            return tree

        op, args = tree[0], tree[1:]
        opname = opnames[op]
        args = tuple(self._simplify_tree(arg) for arg in args)

        if all(is_constant(arg) for arg in args):
            return op(*args)

        opt_fun = self.optimizers.get(opname)
        if opt_fun is not None:
            opt_tree = opt_fun(args)
            if opt_tree is not None:
                return opt_tree
        return op, *args

    def simplify(self):
        simplified = self._simplify_tree(self._p)
        source = self._to_source(simplified)
        new_program = self.from_source(source)
        return new_program

    def __eq__(self, other):
        return self.source == other.source

    def __hash__(self):
        return hash((self._source, self._max_arity))


class GA:
    def __init__(self, config):
        self.conf = config

        self.columns = ()
        self.individuals = ()
        self.steps_taken = 0
        self._max_arity = None
        self.old_scores = dict()
        self.front = ()

    def _from_df(self, X):
        """Transpose the X array into columns as args"""
        Xa = np.array(X)

        if len(Xa.shape) == 1:
            # We have a row vector (such as a Pandas Series). Automatically turn it to a column vector.
            Xa = np.array(Xa, ndmin=2).transpose()

        if len(Xa.shape) > 2:
            raise ValueError(f'Invalid args shape: {Xa.shape}. You need a row per data point.')

        if self._max_arity is not None and Xa.shape[1] != self._max_arity:
            raise ValueError(
                f'Args must have {self._max_arity} cols, but has {Xa.shape[1]}'
            )
        return Xa.transpose()

    def predict(self, X, max_complexity=float('inf')):
        """Predict using the individual with the least training error"""
        errors = ((v[0], i) for i, v in self.old_scores.items() if v[1] <= max_complexity)
        _, best = min(errors, key=lambda e: e[0])
        return best.eval(self._from_df(X))

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.columns = X.columns
        params = self._from_df(X)
        y = np.array(y)
        self._max_arity = len(params)

        p = Program(
            source='0',
            max_arity=self._max_arity,
            columns=self.columns,
            config=self.conf
        )

        self.individuals = tuple(
            p if random.random() < self.conf.zero_program_chance else p.mutate()
            for _ in range(self.conf.n)
        )
        self.old_scores = dict()
        self.steps_taken = 0
        self._step(params, y)

    def fit_partial(self, X, y):
        Xt = self._from_df(X)
        y = np.array(y)
        self._step(Xt, y)

    def _get_random(self, proportion):
        target_size = int(proportion * len(self.individuals))
        target = self.individuals * int(proportion+1)
        return (random.choice(target) for _ in range(target_size))

    def _step(self, Xt, y):
        can_cross_over = self._get_random(self.conf.crossover_children)
        crossed_over = (i.crossover_with_one(self.individuals) for i in can_cross_over)
        can_mutate = self._get_random(self.conf.mutation_children)
        mutated = (i.mutate() for i in can_mutate)

        def perhaps_simplify(i: Program):
            if random.random() < self.conf.simplify_chance:
                return i.simplify()
            else:
                return i

        new_gen = map(perhaps_simplify, chain(crossed_over, mutated))
        new_scores = {i: fitness(i, Xt, y) for i in new_gen if i not in self.old_scores}
        self.old_scores.update(new_scores)

        final, front = nsgaii_cull(self.old_scores, self.conf.n)

        self.steps_taken += 1
        self.front = {ind: self.old_scores[ind] for ind in front}
        self.old_scores = final
        self.individuals = tuple(final)
