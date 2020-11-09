SymReg is a Symbolic Regression library aimed to be easy to use and fast.

You can use it to find an expression that turns inputs into output. The expression can use arbitrary building blocks, not just weighted sums as in linear models.

It uses a modified [NSGA-II](https://ieeexplore.ieee.org/document/996017) algorithm, and applies NumPy functions for vectorized evaluation of input.

## Usage demo

```python
from symreg import Regressor

r = Regressor(duration=5, verbose=True)
X = [[1, 0], [0, 1], [1, 2], [-1, -2]]
y = [.5, .5, 1.5, -1.5]     # We want the average of the arguments

r.fit(X, y)

for score in r.results():
    print(score)
# {'error': 1.1875003470765195, 'complexity': 2, 'program': Program('exp -1.3839406053570065', 2)}
# {'error': 0.25, 'complexity': 3, 'program': Program('neg neg $1', 2)}
# {'error': 0.25, 'complexity': 3, 'program': Program('neg neg $0', 2)}
# {'error': 0.07646678112187726, 'complexity': 4, 'program': Program('div $1 neg -1.3959881664397722', 2)}
# {'error': 0.0, 'complexity': 6, 'program': Program('div add $0 $1 neg -2', 2)}
# {'error': 0.0, 'complexity': 6, 'program': Program('div add $1 $0 neg -2', 2)}

# Program('div add $0 $1 neg -2', 2) means (a + b)/(-(-2)), which is equivalent to (a+b)/2
# It also found an argument-swapped version, and some simpler approximations.

r.predict([4, 6])
# array(5)
# The mean of 4 and 6 is 5. Correct!

r.predict([[4, 6], [1, 2]])
# array([5. , 1.5])
# Also handles vectorized data. Note that a row is a set of parameters.
# Here, $0=4 and $1=6 for the first row, and $0=1 and $1=2 for the second row in the 2d array.

```

You can see more examples of usage in the [Jupyter Notebook file](Metaopt.ipynb).

Install with `pip install symreg`.

## Inspiration

The following projects inspired us:

 * [Eureqa Formulize](http://nutonian.wikidot.com/) - a proprietary piece of 
 otherwise-great software, which is not available anymore
 * [gplearn](https://github.com/trevorstephens/gplearn) - which is Free Software and offers strict `scikit-learn` compatibility (support pipeline and grid search), but does not support multiobjective optimization
 
 Contrary to `gplearn`, we decided to avoid depending on `scikit-learn` for implementation simplicity, but still keep the general API of "fit" and "predict", which is intuitive.
 
 Additionally, we do not perform function closures (patching of infinities). All candidates are passed as-is to NumPy, relying on the fitness function to eliminate numerically unstable ones (an error of infinity or NaN is infinitely penalized).
 
## Technical details

By using a Pareto frontier instead of a single criterion, we can use elitism (keeping the best alive forever), but also maintain genetic diversity.

In the 2D case, we modified NSGA-II to use a faster [O(N*log(N)) Pareto algorithm](https://math.stackexchange.com/a/1937583), because the general N-dimensional algorithm from the paper has poor time complexity. 

We include the density penalty from NSGA-II, which is fast, and helps us further diversify by spreading individuals throughout the frontier.

We do not use NumPy where it is not appropriate. When dealing with lots of small structures, like the scores of a generation which are length 2 each, the line profiler showed a better execution time with plain lists or tuples.

As with many other regression algorithms, we recommend that the input is scaled before training. This is especially true while SymReg does not have a good regression method for constants. 

Still, we personally prefer to avoid changing the sign of data - it makes interpreting the resulting mathematical functions in one's mind more difficult.

## Parameters

We tuned metaparameters according to held-out test error on the Boston dataset (see [the notebook](Metaopt.ipynb) and the bottom of [metaopt.ods](metaopt.ods)). The clearest gradient seems to come from `grow_leaf_mutation_chance`, which should be around 0.4 and not around 0.2. The second-clearest seems to be grow_root_mutation_chance, which should be around 0.3 and not 0.4.
 
 Other analyzed parameters don't give out such a clear signal. The number of individuals in a generation, `n`, is around 65 in both the top 30 individuals, and the worst quartile. The rest seem not to have a clear influence within the tested range. Perhaps this would change if we tested in a different range.

As always, we must take precautions against [overfitting](https://en.wikipedia.org/wiki/Overfitting). Always use a validation set and a test set, especially with such flexible and complex models as Genetic Programming.

While constraining oneself to use simpler expressions can have some [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) effect, **never look at the test set until the end** (and you are sure you won't modify anything else), and only then can you discover the true performance of your algorithm.

## Engineering & Architecture 

We used [**Test-Driven Development**](https://danuker.go.ro/tdd-revisited-pytest-updated-2020-09-03.html) during implementation. We enjoyed it thoroughly.

The "unit" of a unit test is not necessarily a method, but a stable interface, through which the tests must do their job (including testing for detailed behavior). This is in order to keep the implementation flexible; however, testing from too far away might make it harder to debug a failing test.

That does not mean that we allow the tests to be slow. We keep the test suite running in about a second, because it allows us to refactor easily, while being confident the code stays correct. 

Some nondeterministic high-level tests may fail from time to time (`test_symreg`). If they pass at least once since changing your code, the code can produce what they require, and you should think of them as passing.

In addition to TDD, we strive to respect Gary Berhnardt's **Imperative Shell/Functional Core**, and Uncle Bob's **Clean Architecture**. These two are integrated and succintly explained [in the author's pretentiously-named blog post here](https://danuker.go.ro/the-grand-unified-theory-of-software-architecture.html).

These principles let us quickly experiment. We tried 3 domination-frontier algorithms, and decided to keep 2 of them. We barely had to modify the tests, because we tested through the stable, higher-level methods also used by the other classes. We did use the debugger a bit, though.

Running all tests can be done with `python -m pytest`. Running tests on the installed `symreg` is done with just `pytest`. You can make the system pretend it's installed with `python setup.py develop`.

## Next steps

The author wishes to eventually implement the following further features (but pull requests are welcome as well, of course):
    
* Printing a program shows the Pandas column names instead of `$0`, `$1`...
* Add stopping criteria (right now there is only time):
    * stops when the earliest criterion hits: time, number of generations, or stagnation
* Split validation data from training data, early stopping on validation error increase
* Allow choosable fitness function and coding blocks
* Allow `fit_partial` straight from `symreg`
* Multiprocessing (threading is not enough, because we're CPU bound and there is the GIL).
* Implement predict_proba, which polls all the individuals in a population?
* Pretty plots while training
    * Perhaps a UI like Formulize?
* Switch the Program representation to a tree instead of a tuple. This would allow:
    * Better printing of programs (with parentheses, or infix notation, or spreadsheet formulas...)
    * Crossover between individuals
    * Easier reduction/pruning mutations (we currently only have additive and replacement mutations)
        * One type of reduction mutation could be lossless simplification (`add 2 3` -> `5`)
* Evaluation caching 
    * Remember last N programs and their results in a dictionary regardless of score
* Gradient descent for constants
    * Can be implemented as a mutation; but must tune its performance (i.e., how often should it happen, how much time to spend etc.)
* Automated speed and quality tests (currently, we test manually using the notebook and/or profiler).

Feedback is appreciated. Please comment as a GitHub issue, or any other way ([you can contact the author directly here](https://danuker.go.ro/pages/contactabout.html)).
