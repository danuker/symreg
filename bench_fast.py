# This file is used for benchmarking

import random
from time import time

import pandas as pd
from symreg import Regressor

from sklearn.datasets import load_diabetes


def main():
    bunch = load_diabetes()

    X = pd.DataFrame(bunch['data'], columns=bunch['feature_names'])
    X /= X.median()
    y = pd.Series(bunch['target'])

    for i in range(10):
        random.seed(i)
        r = Regressor(generations=11)   # Match GPLearn's error
        start = time()
        r.fit(X, y)
        taken = time()-start
        yhat = r.predict(X)
        print(f'{taken}\t{(abs(yhat-y)).mean()}')


main()
