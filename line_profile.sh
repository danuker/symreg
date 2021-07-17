#!/bin/bash
# Line profiler: find out what lines are slow
# Need to decorate desired functions with @profile beforehand; not so convenient.

# Shoululd use viztracer instead! Much nicer output.
# pip install viztracer
# viztracer tests/test_regressor.py

rm "$1.lprof"
kernprof -l "$1"
python -m line_profiler "$1.lprof"

rm "$1.lprof" # Cleanup when done
