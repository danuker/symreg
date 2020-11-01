#!/bin/bash
# Run mutation testing on the source code
# to verify tests

mutmut run --paths-to-mutate ga.py
mutmut show

