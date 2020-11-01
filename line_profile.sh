#!/bin/bash
# Line profiler: find out what lines slow
# Need to decorate desired functions with @profile beforehand

rm ga.py.lprof
kernprof -l ga.py
python -m line_profiler ga.py.lprof


