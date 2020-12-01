# This file is used for benchmarking

import random
from time import time

import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.linear_model import LinearRegression

from symreg import Regressor


from sklearn.datasets import load_diabetes


def main():
    bunch = load_diabetes()

    X = pd.DataFrame(bunch['data'], columns=bunch['feature_names'])
    X -= X.mean()
    X /= X.std()
    y = pd.Series(bunch['target'])
    y -= y.mean()
    y /= y.std()
    times = np.logspace(-3, 0, 20) * 19

    regs = ['gplearn', 'linreg', 'symreg']
    # regs = ['symreg']

    for regname in regs:
        print()
        print(regname)
        for i, t in enumerate(times):
            regmap = {
                'symreg': Regressor(duration=t),
                'gplearn': SymbolicRegressor(generations=int(t)+1),
                'linreg': LinearRegression()
            }

            random.seed(i)
            r =regmap[regname]
            start = time()
            r.fit(X, y)
            taken = time()-start
            yhat = r.predict(X)
            print(f'{taken}\t{(abs(yhat-y)).mean()}')

        if regname == 'symreg':
            print('\n'.join(str(res) for res in r.results()))
            print("Features:")
            print("age, age in years")
            print("sex")
            print("bmi, body mass index")
            print("bp, average blood pressure")
            print("s1 tc, T-Cells (a type of white blood cells)")
            print("s2 ldl, low-density lipoproteins")
            print("s3 hdl, high-density lipoproteins")
            print("s4 tch, thyroid stimulating hormone")
            print("s5 ltg, lamotrigine")
            print("s6 glu, blood sugar level")
            print("Target: Column 11 is a quantitative measure of disease progression one year after baseline")
            # Seems like this program is quite good:
            # mul 0.4648468601443117 add $bmi $s5
            # or BMI and lamotrigine being equal-importance and predicting diabetes on increase

main()
