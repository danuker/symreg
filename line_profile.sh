#!/bin/bash
# Line profiler: find out what lines slow
# Need to decorate desired functions with @profile beforehand

rm symreg.py.lprof
kernprof -l symreg.py
python -m line_profiler symreg.py.lprof


