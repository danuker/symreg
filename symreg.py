from ga import Program, GA


class Regressor:
    def __init__(self, n=50, steps=50):
        self._ga = GA(n, steps)

    def fit(self, X, y):
        self._ga.fit(X, y)

    def fit_partial(self, X, y):
        self._ga.fit_partial(X, y)

    def predict(self, X):
        y_pred = self._ga.predict(X)
        return y_pred

    def individual_train_scores(self):
        return self._ga.old_scores


if __name__ == '__main__':
    import pytest

    pytest.main(['test_symreg.py', '--color=yes'])
