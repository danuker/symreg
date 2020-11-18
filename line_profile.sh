#!/bin/bash
# Line profiler: find out what lines slow
# Need to decorate desired functions with @profile beforehand

rm "$1.lprof"
kernprof -l "$1"
python -m line_profiler "$1.lprof"
