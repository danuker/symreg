class Regressor:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[:][1]


if __name__ == '__main__':
    import pytest

    pytest.main(['test_symreg.py'])
