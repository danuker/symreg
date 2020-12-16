from dataclasses import dataclass
from time import time

from symreg.ga import GA


@dataclass(frozen=True)
class Configuration:
    n: int = 50
    zero_program_chance: float = 0.5
    hoist_mutation_chance: float = .24
    prune_mutation_chance: float = .16
    grow_root_mutation_chance: float = .24
    grow_leaf_mutation_chance: float = .16
    complete_tree_as_new_subtree_chance: float = .5
    mutation_children: float = .7
    crossover_children: float = .7
    simplify_chance: float = .1
    int_std: float = 3
    float_std: float = 4


class Regressor:
    def __init__(
            self,
            n=50,
            duration=float('inf'),
            generations=float('inf'),
            stagnation_limit=float('inf'),
            verbose=False,
            zero_program_chance=0.5,
            hoist_mutation_chance=.24,
            prune_mutation_chance=.16,
            grow_root_mutation_chance=.24,
            grow_leaf_mutation_chance=.16,
            complete_tree_as_new_subtree_chance=.5,
            mutation_children=.7,
            crossover_children=.7,
            simplify_chance=.1,
            int_std=3,
            float_std=4,
    ):
        self.conf = Configuration(
            n=n,
            zero_program_chance=zero_program_chance,
            hoist_mutation_chance=hoist_mutation_chance,
            prune_mutation_chance=prune_mutation_chance,
            grow_root_mutation_chance=grow_root_mutation_chance,
            grow_leaf_mutation_chance=grow_leaf_mutation_chance,
            complete_tree_as_new_subtree_chance=complete_tree_as_new_subtree_chance,
            mutation_children=mutation_children,
            crossover_children=crossover_children,
            simplify_chance=simplify_chance,
            int_std=int_std,
            float_std=float_std,
        )

        self._ga = GA(config=self.conf)

        self.duration = duration
        self.verbose = verbose
        self.columns = ()
        self.training_details = {'steps': 0, 'duration': 0}
        self.steps_to_take = generations
        self.max_stagnation_generations = stagnation_limit
        self._last_results = {}
        self._stagnation = 0

    def fit(self, X, y):
        start = time()
        self._ga.fit(X, y)
        last_printed = time()
        taken = time() - start

        while self.can_continue(taken):
            taken = time() - start
            if self.verbose and time() - last_printed > 1:
                last_printed = time()
                print(f'Time left  : {(self.duration - taken):.2f}s')
                print(f'Best so far: {min(s for s in self._ga.old_scores.values())} (error, complexity)')

            self._ga.fit_partial(X, y)

            new_results = self.results()
            if new_results != self._last_results:
                self._stagnation = 0
                self._last_results = new_results
            else:
                self._stagnation += 1

            self.training_details = {
                'generations': self._ga.steps_taken,
                'stagnated_generations': self._stagnation,
                'duration': time() - start,
            }

        if self.verbose:
            print(f'Complete. {self.training_details}')

    def can_continue(self, taken):
        return taken < self.duration and \
               self._ga.steps_taken < self.steps_to_take and \
               self._stagnation < self.max_stagnation_generations

    def predict(self, X, max_complexity=float('inf')):
        return self._ga.predict(X, max_complexity)

    def results(self):
        scores = self._ga.front
        front = [
            {'error': score[0], 'complexity': score[1], 'program': program}
            for program, score in scores.items()
        ]
        return sorted(front, key=lambda s: s['complexity'])
