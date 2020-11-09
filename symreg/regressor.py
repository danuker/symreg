from time import time

from symreg.ga import GA
from symreg.nsgaii import ndim_pareto_ranking, SolutionScore


class Regressor:
    def __init__(
            self,
            n=50,
            duration=5,
            verbose=False,
            zero_program_chance=0.5,
            grow_root_mutation_chance=.3,
            grow_leaf_mutation_chance=.4,
            int_std=3,
            float_std=4,
    ):
        self._ga = GA(
            n=n,
            steps=1,
            zero_program_chance=zero_program_chance,
            grow_root_mutation_chance=grow_root_mutation_chance,
            grow_leaf_mutation_chance=grow_leaf_mutation_chance,
            int_std=int_std,
            float_std=float_std,
        )
        self.duration = duration
        self.verbose = verbose

    def fit(self, X, y):
        start = time()
        self._ga.fit(X, y)

        taken = time() - start
        last_printed = start
        while taken < self.duration:
            if self.verbose and time() - last_printed > 5:
                last_printed = time()
                print(f'Time left  : {int(self.duration - taken + .9)}s')
                print(f'Best so far: {min(s for s in self._ga.old_scores.values())} (error, complexity)')

            self._ga.fit_partial(X, y)

            taken = time() - start

        if self.verbose:
            print(f'Generations evolved: {self._ga.steps_taken}')

    def predict(self, X):
        y_pred = self._ga.predict(X)
        return y_pred

    def results(self):
        scores = self._ga.old_scores
        scores = SolutionScore.scores_from_dict(scores)
        front = ndim_pareto_ranking(scores)[1]
        front = [{'error': ss.scores[0], 'complexity': ss.scores[1], 'program': ss.individual} for ss in front]
        return sorted(front, key=lambda s: s['complexity'])

