import numpy as np

from ga import Program, GA


class Regressor:
    def __init__(self):
        pass

    def fit(self, X, y):
        args = X
        max_arity = len(args[0])
        res = np.array(y)

        self.ga = GA({
            Program(['$0'], max_arity),
        })
        y_pred = self.ga.evaluate(X)
        print('real:', y)
        print('predicted:', np.array(y_pred))

        raise ValueError

    def predict(self, X):
        y_pred = self.ga.evaluate(X)
        print('predicted:', np.array(y_pred))
        return y_pred


if __name__ == '__main__':
    import pytest

    pytest.main(['test_symreg.py'])
