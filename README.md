Symreg is a Symbolic Regression library aimed to be easy to use and fast.

It uses the [fast non-dominated sort from NSGA-II](https://ieeexplore.ieee.org/document/996017) and applies pandas/numpy functions for speedy, 
vectorized evaluation.

## Usage

Example taken from [`test_symreg.py`](test_symreg.py):

```python
from symreg import Regressor

s = Regressor()
X = [[0, 0], [0, 1], [0, 2]]
y = [    0,      1,      2]

s.fit(X, y)
assert s.predict([0, 3]) == 3
```

## Inspiration

The following projects inspired me:

 * [Eureqa Formulize](http://nutonian.wikidot.com/), a proprietary piece of 
 otherwise-great software
 * [gplearn](https://github.com/trevorstephens/gplearn), which does not support multiobjective optimization, and which requires complexity for `scikit-learn` compatibility (support for pipeline and grid search).
 
 Contrary to `gplearn`, we decided to avoid the `scikit-learn` dependency, but still keep the general design of "fit" and "predict", which are intuitive.
 
 Also, we do not perform function closures. All candidates are passed as-is to numpy, relying on the fitness function to eliminate numerically unstable ones (a fitness of infinity or NaN is accounted for).
 
## Technical details

We use Pareto-efficiency for selecting individuals: instead of "best" meaning a single individual, "best" means all individuals with the least error for every given level of complexity/speed ([Pareto efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency)).

This allows both:
* elitism (allowing the best to survive), and
* maintaining genetic diversity (the number of individuals on the Pareto frontier is `O(n_objectives - 1)`).

While NSGA-II also includes a density measure, we forego that for now, for simplicity.

#### Engineering & Architecture 

This implementation follows [**Test-Driven Development**](https://danuker.go.ro/tdd-revisited-pytest-updated-2020-09-03.html). TDD promises [faster development]() and a minimalist design. 

The "unit" of a unit test is not a method, but a class which has publically-usable interface, through which the tests must do their job (including testing for detailed behavior). This is in order to keep the implementation flexible.

That does not mean that we allow the tests to be slow. We keep the test suite running in under one second, because it allows us to refactor easily, while being confident the code is kept correct.

In addition, we strive to respect Gary Berhnardt's **Imperative Shell/Functional Core**, and Uncle Bob's **Clean Architecture**. These two are integrated and succintly explained [in the author's pretentiously-named blog post](https://danuker.go.ro/the-grand-unified-theory-of-software-architecture.html). The author is attempting to [dog-food](https://en.wikipedia.org/wiki/Eating_your_own_dog_food) this architectural style.
